# Guida al codice di RLX

Questa guida spiega **ogni parte** del progetto: cosa fa, come è scritta e soprattutto *perché* è scritta così. L'idea è che tu la legga con il codice aperto accanto, file per file, nell'ordine dei capitoli. Alla fine trovi una sezione "Il tuo percorso" con l'ordine in cui implementare gli algoritmi.

---

## 0. La mappa del progetto

```
RLX/
├── rlx/                        il pacchetto Python
│   ├── core/                   le interfacce (i "contratti")
│   │   ├── env.py              Env, DiscreteEnv, StepResult
│   │   └── agent.py            Agent, TabularAgent, PlanningAgent, Transition
│   ├── envs/                   gli ambienti
│   │   ├── grid.py             GridEnv: una griglia generica costruita da testo
│   │   ├── classic.py          GridWorld, FrozenLake, CliffWalking
│   │   └── __init__.py         registro ENVS + make_env / make_custom_env
│   ├── algorithms/             gli algoritmi
│   │   ├── dp/                 Value Iteration (completo), Policy Eval/Iteration (stub)
│   │   ├── mc/                 Monte Carlo control (stub)
│   │   ├── td/                 SARSA, Q-learning, Expected SARSA, n-step SARSA (stub)
│   │   ├── planning/           Dyna-Q (stub)
│   │   ├── bandits/            testbed a 10 braccia (completo) + agenti (stub)
│   │   ├── baselines/          RandomAgent
│   │   └── __init__.py         registro ALGORITHMS + make_agent
│   ├── training/
│   │   ├── loop.py             train, run_episode, evaluate
│   │   └── callbacks.py        Callback, History, PrintLogger, EarlyStopping...
│   └── dashboard/
│       ├── server.py           FastAPI: pagina, catalogo, WebSocket
│       ├── session.py          una connessione = una Session con il suo thread di training
│       └── static/             index.html, style.css, app.js (niente framework)
├── tests/                      pytest
├── examples/value_iteration.py
├── pyproject.toml              dipendenze, configurazione di pytest e ruff
└── .github/workflows/tests.yml CI: ruff + pytest a ogni push
```

Il principio che tiene insieme tutto è la **separazione delle responsabilità**:

- l'**ambiente** sa solo come evolve il mondo;
- l'**agente** sa solo come scegliere azioni e come imparare;
- il **training loop** sa solo come farli parlare tra loro;
- le **callback** osservano il training senza modificarlo;
- la **dashboard** è un cliente come un altro: usa esattamente la stessa funzione `train` che useresti da uno script.

Grazie a questo, quando scriverai Q-learning dovrai toccare **un solo metodo** di **un solo file**, e lo vedrai subito funzionare nella dashboard e nei test.

### Come si avvia

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # -e = "editable": le modifiche al codice sono subito attive
pytest                          # 33 test passano, 8 sono "skipped" (gli stub)
python -m rlx.dashboard         # poi apri http://127.0.0.1:8000
```

`pip install -e` legge `pyproject.toml`. Lì le dipendenze sono divise in gruppi: il nucleo richiede solo `numpy`; la dashboard aggiunge `fastapi` e `uvicorn`; `dev` aggiunge `pytest`, `httpx` (serve al client di test di FastAPI) e `ruff` (linter e formatter). Così chi vuole solo la libreria non è costretto a installare un web server.

---

## 1. `rlx/core/env.py`: l'interfaccia dell'ambiente

### 1.1 `StepResult`

```python
@dataclass(frozen=True)
class StepResult:
    state: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
```

**Perché una dataclass e non una tupla?** `s, r, d, _ = env.step(a)` funziona, ma è fragile: se scambi due campi nessuno se ne accorge. Con `step.reward` il codice si legge da solo. `frozen=True` la rende immutabile: un risultato di step è un fatto avvenuto, nessuno deve poterlo modificare dopo.

**Perché `terminated` e `truncated` separati?** È una distinzione che conta davvero per la correttezza degli algoritmi (è la stessa scelta fatta da Gymnasium):

- `terminated`: si è raggiunto uno stato terminale dell'MDP (goal, buca). Il valore futuro è **zero**, quindi il target TD è solo `r`.
- `truncated`: ci siamo fermati per un motivo *esterno* all'MDP, cioè il limite di passi. Lo stato **non** è terminale, quindi il target deve ancora fare bootstrap: `r + γ·max Q(s',·)`.

Se le confondi, l'agente impara che "girare a vuoto per 200 passi" porta a uno stato di valore 0, il che è falso. La property `done` esiste solo per comodità nel loop ("l'episodio è finito per qualunque motivo?").

`field(default_factory=dict)` serve perché in Python un default mutabile (`info: dict = {}`) sarebbe **condiviso** tra tutte le istanze: un classico bug.

### 1.2 `Env` (classe astratta)

```python
class Env(ABC):
    n_actions: int
    @abstractmethod
    def reset(self, seed: int | None = None) -> int: ...
    @abstractmethod
    def step(self, action: int) -> StepResult: ...
```

È il contratto minimo. `ABC` + `@abstractmethod` fanno sì che Python si rifiuti di istanziare una sottoclasse che non implementa questi metodi: l'errore arriva subito, non a metà training. L'interfaccia è volutamente piccola: più è piccola, più ambienti diversi (anche non tabulari, in futuro: CartPole, ambienti multi-agente) potranno rispettarla.

### 1.3 `DiscreteEnv`: un MDP finito con modello

Il modello di transizione ha la forma che ti è familiare da Sutton & Barto:

```python
P[s][a] = [(prob, next_state, reward, done), ...]
```

cioè, per ogni coppia (s, a), la lista dei possibili esiti con le loro probabilità. Ho scelto questa forma (invece di una matrice `T[s, a, s']`) perché:

1. rappresenta naturalmente **reward diverse per esiti diversi** dalla stessa (s, a), cosa che una matrice `T` da sola non fa;
2. è **sparsa**: in una griglia ogni (s, a) ha 1-3 esiti, non `n_states`;
3. corrisponde 1:1 alla notazione `p(s', r | s, a)` del libro.

Quando serve la forma matriciale c'è `transition_matrix()` (e `expected_reward()` per `R[s, a]`), usati nei test.

**La scelta di design più importante del file** è questa:

```python
def step(self, action):
    outcomes = self.P[self.state][action]
    probs = [o[0] for o in outcomes]
    idx = self.rng.choice(len(outcomes), p=probs) if len(outcomes) > 1 else 0
    _, next_state, reward, terminated = outcomes[idx]
```

`step` **campiona dal modello stesso**. Non esiste una seconda implementazione della dinamica. Così è impossibile che Value Iteration (che legge `P`) e Q-learning (che usa `step`) vedano due mondi diversi. Se ci fosse un bug in `P`, entrambi lo vedrebbero, e i test lo troverebbero.

Altri dettagli:

- `self.rng = np.random.default_rng(seed)`: ogni ambiente ha il **suo** generatore. Non si usa mai `np.random.seed(...)` globale, perché un generatore globale fa sì che l'agente e l'ambiente si "rubino" numeri casuali a vicenda e la riproducibilità salta appena cambi qualcosa.
- `reset(seed=...)` reinizializza il generatore solo se gli passi un seed; altrimenti continua lo stream. Questo permette "seed al primo episodio, poi continua" (vedi il loop).
- `max_steps` e il contatore `self.t` implementano il troncamento in un unico punto: ogni ambiente discreto lo eredita gratis.
- `_validate()` controlla che ogni `P[s][a]` sommi a 1 e che la distribuzione iniziale sommi a 1. Fallire subito con un messaggio chiaro (`P[3][1] probabilities sum to 0.9`) ti risparmia ore di debug.

---

## 2. `rlx/core/agent.py`: l'interfaccia dell'agente

### 2.1 `Transition`

`Transition(state, action, reward, next_state, terminated, truncated)` è la tupla (S_t, A_t, R_{t+1}, S_{t+1}) del libro, più i due flag di fine episodio. È ciò che l'agente riceve dopo ogni passo.

### 2.2 `Agent`: gli hook

Il ciclo di vita che il loop impone a ogni agente è:

```
begin_episode(s0)
ripeti:  a = act(s)  →  env.step(a)  →  observe(transition)
end_episode()
```

**Perché questi quattro hook e non un unico `update()`?** Perché devono andare bene a *tutti* gli algoritmi tabulari del libro, che imparano in momenti diversi:

| Algoritmo | Dove impara |
|---|---|
| SARSA, Q-learning, Expected SARSA | in `observe`, a ogni passo |
| Monte Carlo | accumula in `observe`, impara in `end_episode` (serve il ritorno completo G) |
| n-step SARSA | buffer scorrevole in `observe`, svuota il buffer in `end_episode` |
| Dyna-Q | in `observe`: aggiornamento reale + modello + n passi di planning |

`begin_episode`, `observe` e `end_episode` hanno un'implementazione vuota di default, quindi un agente sovrascrive solo quelli che gli servono. `act` invece è astratto: ogni agente deve saper scegliere un'azione.

`act(state, greedy=False)`: il flag `greedy` separa la **policy di comportamento** (esplorativa, usata in training) dalla **policy target** (greedy, usata in valutazione e nel bottone "Watch greedy"). È la distinzione on-policy/off-policy resa esplicita nell'API.

`values()`, `policy()` e `q_values()` sono metodi di **introspezione**: servono alla dashboard per disegnare V(s), le frecce della policy e i triangoli di Q(s,a). Non sono usati dall'apprendimento.

Gli attributi di classe (`name`, `kind`, `implemented`, `reference`, `default_params`) sono **metadati**. Il registro e la dashboard li leggono per costruire i menu e il form degli iperparametri. Usare `ClassVar` dice a chi legge (e al type checker) che appartengono alla classe, non alla singola istanza.

### 2.3 `TabularAgent`: ciò che hanno in comune gli algoritmi di controllo

Quasi tutti gli algoritmi di controllo tabulari hanno: una tabella `Q[s, a]`, una policy ε-greedy, γ, α. Differiscono solo nella **regola di aggiornamento**. Allora tutto il resto sta in una classe base, e le sottoclassi implementano solo `observe` / `end_episode`. È il principio DRY ("don't repeat yourself") applicato con giudizio: si fattorizza ciò che è *davvero* comune.

Un dettaglio che conta più di quanto sembri:

```python
def greedy_action(self, state):
    q = self.Q[state]
    best = np.flatnonzero(q == q.max())
    return int(self.rng.choice(best))
```

**Rottura dei pareggi casuale.** All'inizio Q è tutta zeri. `np.argmax` restituirebbe sempre l'azione 0 (su), quindi l'agente "greedy" andrebbe sempre su e l'esplorazione dipenderebbe solo da ε. Su CliffWalking questo rallenta moltissimo l'apprendimento. Scegliere a caso tra gli ottimi è la cosa corretta.

`policy()` restituisce le probabilità ε-greedy esatte: `ε/|A|` per ogni azione, più `1-ε` sulla greedy. Ti servirà in Expected SARSA, dove il target è `Σ_a π(a|s') Q(s', a)`.

### 2.4 `PlanningAgent`: programmazione dinamica

Gli algoritmi DP non interagiscono con l'ambiente: leggono `P`. Per questo `PlanningAgent.__init__` riceve l'**ambiente** (non solo le dimensioni), e invece di `observe` hanno `plan_step()`, che fa un'iterazione (es. uno sweep) e restituisce `max |ΔV|`, usato dal loop per capire quando fermarsi.

`q_from_v(V)` è il **lookahead a un passo**, il mattone di tutta la DP:

```python
Q[s, a] = sum(p * (r + gamma * V[s2] * (not done)) for p, s2, r, done in P[s][a])
```

Nota `(not done)`: se l'esito è terminale il valore futuro è 0. Gli stati terminali stessi sono definiti come assorbenti con reward 0 (vedi `grid.py`), quindi il loro V resta 0 automaticamente.

Infine nota l'asimmetria voluta in `make_agent` (cap. 5): agli agenti `planning` si passa l'ambiente, agli agenti `online` **solo** `n_states` e `n_actions`. Un agente model-free non può "barare" leggendo `P` nemmeno per sbaglio: l'API non glielo permette.

---

## 3. `rlx/envs/grid.py`: una griglia generica

Invece di scrivere tre ambienti separati, ho scritto **un** ambiente a griglia parametrizzato da un layout testuale, e i tre classici sono solo configurazioni. Lo stesso meccanismo alimenta l'editor della dashboard: le griglie che disegni sono `GridEnv` come gli altri.

```
S partenza   . libero   # muro   G goal (terminale)   H buca (terminale)   C dirupo
```

### 3.1 Stati e coordinate

`s = riga * n_cols + colonna` (ordine row-major). **Anche i muri sono stati**, solo irraggiungibili. Sembra uno spreco, ma rende banale la mappa stato→cella in tutto il codice e nel frontend (`r = floor(s / cols)`, `c = s % cols`). Con poche decine di stati il costo è nullo.

### 3.2 Costruzione del modello

`_build_model` chiama `_outcomes(s, a)` per ogni coppia. La logica è divisa in due livelli:

- `_move(state, action)`: l'effetto **deterministico** di un movimento. Se esci dalla griglia o colpisci un muro resti dove sei; se entri in G o H l'episodio termina; se entri in C prendi la penalità e torni alla partenza **senza** terminare (esattamente l'Esempio 6.6 del libro).
- `_outcomes(state, action)`: aggiunge la **stocasticità**. Con probabilità `1 - slip` fai il movimento voluto, e la probabilità `slip` si divide a metà tra le due direzioni perpendicolari. Con `slip = 2/3` ottieni 1/3 per ciascuna delle tre: è la dinamica classica di FrozenLake.

Dettaglio: esiti identici vengono **fusi** (il dizionario `merged`). Esempio: in un angolo, andare su e andare a sinistra ti lasciano entrambi fermo; invece di due esiti `(1/3, s, 0, False)` ne hai uno `(2/3, s, 0, False)`. Non cambia la matematica, ma rende `P` più piccolo e più leggibile nei test.

Gli stati terminali (e i muri) sono **assorbenti**: `P[s][a] = [(1.0, s, 0.0, True)]`. Questo fa uscire `V(terminale) = 0` direttamente dall'equazione di Bellman, senza casi speciali negli algoritmi.

`GridRewards` è una dataclass immutabile con le reward per "entrare in" ciascun tipo di cella. Separarla dal layout permette di riusare lo stesso layout con reward diverse (es. un GridWorld con reward sparse).

### 3.3 Parsing e validazione

`_parse` rifiuta layout vuoti, righe di lunghezza diversa e caratteri sconosciuti, e il costruttore rifiuta layout senza `S`. Dato che i layout arrivano anche dal browser, validare all'ingresso è obbligatorio.

### 3.4 `classic.py`

Tre sottoclassi che fissano layout e reward:

- **GridWorld**: labirinto 5×5 deterministico, −1 a passo, +10 al goal. Percorso ottimo: 8 passi, ritorno 3.
- **FrozenLake**: mappe 4×4 e 8×8 originali, scivoloso. +1 al goal, 0 altrove. Con γ = 1 la probabilità di successo ottima dalla partenza è ≈ 0.8235 (lo verifica un test).
- **CliffWalking**: 4×12, −1 a passo, −100 sul dirupo. Ottimo: 13 passi lungo il bordo, ritorno −13.

### 3.5 Il registro (`envs/__init__.py`)

`ENVS` è un dizionario `id → EnvSpec(id, label, description, factory)`. La dashboard elenca *esattamente* ciò che è registrato. Aggiungere un ambiente = una riga. `make_custom_env` costruisce un `GridEnv` da un layout dell'editor.

---

## 4. `rlx/algorithms/dp/value_iteration.py`: l'algoritmo di riferimento

È l'unico algoritmo completo (oltre al `RandomAgent`), apposta: serve a far funzionare tutto end-to-end e a mostrarti lo stile da seguire.

```python
def plan_step(self) -> float:
    Q = self.q_from_v(self.V)
    new_V = Q.max(axis=1)
    delta = float(np.max(np.abs(new_V - self.V)))
    self.V = new_V
    self.pi = self._greedy_policy(Q)
    return delta
```

È l'operatore di ottimalità di Bellman applicato a tutti gli stati: `V(s) ← max_a Σ p(s',r|s,a)[r + γV(s')]`.

**Sweep sincrono vs in-place.** Lo pseudocodice del libro aggiorna V *in place* (stile Gauss-Seidel: gli stati aggiornati prima sono già usati dagli stati dopo). Qui uso la versione **sincrona** ("a due array": tutti i Q calcolati dal V vecchio). Converge comunque allo stesso V*; l'ho scelta perché ogni sweep è un'applicazione pulita dell'operatore, il che è più facile da ragionare e da visualizzare nella dashboard (vedi i valori "propagarsi" dal goal di un passo per sweep). Un bell'esercizio: implementa la versione in-place e confronta il numero di sweep.

`_greedy_policy` estrae la policy greedy e, in caso di **pareggi**, divide la probabilità tra le azioni ottime (`np.isclose` invece di `==` per robustezza numerica). Ecco perché nella dashboard su CliffWalking vedi due frecce in alcune celle: andare a destra o in basso è davvero equivalente lì.

La policy viene estratta **a ogni sweep** (e non solo alla fine come nel libro) perché la dashboard possa mostrare come cambia, e perché il loop possa valutarla e disegnare la curva di apprendimento.

---

## 5. Gli stub e il registro degli algoritmi

Ogni stub è una classe vera con docstring che spiega cosa implementare, riferimento al paragrafo del libro, e `implemented = False`. I metodi da scrivere lanciano `NotImplementedError`.

- `dp/policy_iteration.py`: `PolicyEvaluation` (valuta la policy uniforme) e `PolicyIteration`.
- `mc/mc_control.py`: Monte Carlo first-visit on-policy.
- `td/td_control.py`: SARSA, Q-learning, Expected SARSA. La docstring del modulo mostra che differiscono **solo nel target**.
- `td/n_step.py`: n-step SARSA con un `deque` come buffer.
- `planning/dyna_q.py`: Dyna-Q con il suo modello appreso `self.model`.
- `bandits/bandits.py`: il testbed a 10 braccia è completo; ε-greedy, UCB e gradient bandit sono da scrivere. I bandit hanno un solo stato, quindi hanno un'interfaccia propria più semplice e non compaiono nella dashboard a griglia.

Una nota su SARSA: è on-policy, quindi il target usa `A_{t+1}`, l'azione che l'agente farà *davvero* dopo. Ma il loop chiama `act(s')` **dopo** `observe`. La soluzione suggerita nella docstring: in `observe` scegli tu `a'`, salvala in `self.next_action`, e restituiscila alla chiamata successiva di `act`. Così l'aggiornamento e il comportamento usano la stessa azione.

`algorithms/__init__.py` contiene `ALGORITHMS` (ordinati come i capitoli del libro) e `make_agent(algo_id, env, seed, **params)`, che:

1. unisce i `default_params` della classe con quelli passati;
2. **rifiuta** parametri sconosciuti (un typo come `epsilom=0.1` deve essere un errore, non essere ignorato in silenzio);
3. passa l'env agli agenti `planning`, solo le dimensioni agli agenti `online`.

---

## 6. `rlx/training/`: il loop e le callback

### 6.1 `run_episode`

Esegue un episodio: `reset`, poi `act → step → observe` finché `done`. Con `learn=False` l'agente agisce in modo greedy e **non** riceve `observe`/`end_episode`: è la valutazione pura. Restituisce un `EpisodeLog` con ritorno, lunghezza e la lista degli stati visitati (la dashboard la disegna come scia arancione).

### 6.2 `train`

```python
def train(env, agent, n_episodes=500, callbacks=(), *, max_iterations=1000, tol=1e-8,
          eval_episodes=0, seed=None) -> Run:
```

Il `*` rende gli argomenti successivi **keyword-only**: `train(env, agent, 500, [], 1000, 1e-8)` non è permesso, devi scrivere `tol=1e-8`. È una scelta di leggibilità per gli argomenti che si confondono facilmente.

`train` guarda il tipo di agente e smista:

- **online** → `_learn`: `n_episodes` episodi, callback `on_episode_end` dopo ognuno.
- **planning** → `_plan`: `plan_step()` fino a `delta < tol` o `max_iterations`; se `eval_episodes > 0`, dopo ogni sweep valuta la policy greedy, così anche la DP ha una "curva di apprendimento" confrontabile con gli altri.

`Run` è lo stato condiviso del training (env, agent, contatori, `stop_requested`). È passato a ogni callback, ed è così che una callback può **fermare** il training senza conoscere il loop: imposta `run.stop_requested = True` e il loop esce al controllo successivo.

### 6.3 Le callback

Il pattern è quello di Keras/Lightning: una classe base con hook vuoti (`on_train_begin`, `on_episode_end`, `on_iteration_end`, `on_train_end`) e callback concrete che sovrascrivono solo quello che serve.

- `CallbackList` applica il **pattern composite**: è una callback che inoltra a una lista di callback, così il loop ne vede sempre una sola.
- `History`: tiene tutti i log in memoria (`history.returns` ti dà l'array dei ritorni da plottare).
- `PrintLogger`: stampa la media mobile ogni N episodi.
- `EarlyStopping`: ferma il training quando la media mobile raggiunge un target.
- `LambdaCallback`: trasforma funzioni qualsiasi in callback, comodo nei notebook.

**Perché le callback e non dei `print` nel loop?** Perché così il loop non cambia mai. La dashboard, i test, uno script con TensorBoard o Weights & Biases: ognuno aggiunge la sua callback. Il loop resta 30 righe leggibili.

`from __future__ import annotations` e `if TYPE_CHECKING:` in `callbacks.py` evitano un **import circolare**: `callbacks.py` cita `Run` (definito in `loop.py`) solo nelle annotazioni di tipo, e `loop.py` importa `callbacks.py`. Con `TYPE_CHECKING` l'import avviene solo per il type checker, mai a runtime.

---

## 7. La dashboard

### 7.1 Architettura

```
Browser (app.js) ──WebSocket JSON──► server.py ──► Session ──► thread: train(env, agent, [StreamCallback])
      ▲                                                                       │
      └──────────────── messaggi "episodes" / "iteration" / "done" ◄──────────┘
```

**Perché WebSocket e non richieste HTTP?** Il training produce un flusso continuo di aggiornamenti dal server verso il browser, e il browser deve poter mandare comandi in qualunque momento (stop, cambio velocità, mosse del giocatore). Una connessione bidirezionale persistente è lo strumento giusto.

### 7.2 `server.py`

- `GET /` serve `index.html`, `/static/*` serve CSS e JS, `GET /api/catalog` restituisce ambienti e algoritmi (con metadati e iperparametri di default) letti dai registri.
- `WS /ws`: per ogni scheda del browser crea una `Session`.

Il punto delicato è che il training gira in un **thread** separato (è normale codice sincrono: la stessa `train` degli script), mentre il server è **asyncio**. Le code `asyncio.Queue` non sono thread-safe, quindi il thread non può fare `outbox.put_nowait(msg)` direttamente. Si usa:

```python
def emit(msg):
    loop.call_soon_threadsafe(outbox.put_nowait, msg)
```

che chiede all'event loop di eseguire la `put` nel suo thread. Un task `sender()` svuota la coda e invia i messaggi. I messaggi in arrivo dal browser sono gestiti con `asyncio.to_thread(session.handle, msg)` perché `stop()` può bloccare (aspetta la fine del thread con `join`) e non vogliamo bloccare l'event loop. Nel `finally`, alla disconnessione, si ferma il training e si cancella il sender: nessun thread orfano.

### 7.3 `session.py`

`Session.handle` è un **dispatcher**: un dizionario `tipo messaggio → metodo` (`start`, `stop`, `speed`, `rollout`, `play_reset`, `play_step`). Qualunque eccezione viene trasformata in un messaggio `{"type": "error"}`: un input sbagliato dal browser non deve mai far cadere la connessione. Lanciare uno stub produce "Not implemented yet: QLearning.observe" nella barra di stato.

`StreamCallback` è il ponte tra il training e il browser, ed è l'esempio perfetto del perché le callback esistono: la dashboard non ha modificato una riga del loop. Due accorgimenti:

1. **Rate limiting**: senza ritardo, Value Iteration o Q-learning su una griglia fanno migliaia di episodi al secondo; il browser non può disegnarne più di ~60. Quindi, se il ritardo è 0, si inviano al massimo 30 snapshot al secondo, e i ritorni degli episodi intermedi vengono **accumulati** e inviati in blocco (`returns: [...]`), così la curva non perde nessun punto.
2. **Velocità e stop**: `_pace` dorme `delay_s` (modificabile dallo slider *durante* il training) e controlla `stop_event`, un `threading.Event`: il modo standard e thread-safe per dire a un thread di fermarsi.

`snapshot(agent)` prende V, π e Q, li arrotonda a 4 decimali e sostituisce `inf`/`nan` (JSON non li supporta).

`rollout` esegue un episodio greedy dell'agente addestrato ("Watch greedy"); `play_reset`/`play_step` gestiscono un ambiente separato in cui **giochi tu** con le frecce.

### 7.4 Il frontend (`static/`)

Niente React, niente build, niente CDN: HTML + CSS + JavaScript puro. Funziona offline, si legge in un pomeriggio e non invecchia.

`app.js` segue un principio semplice: **flusso dati unidirezionale**. I messaggi del server aggiornano l'oggetto `state`; poi `render()` ridisegna *tutto* da `state`. Nessun'altra funzione disegna. È la stessa idea di React, senza React: il disegno è una funzione dello stato, quindi non ci sono mai disegni "vecchi" incoerenti.

- `drawGrid`: colora ogni cella in base a V(s) (blu positivo, rosso negativo, intensità proporzionale a |V|/max|V|), oppure in modalità Q disegna 4 triangoli per cella (uno per azione, sul lato verso cui punta), poi le frecce della policy (tutte le azioni ottime in caso di pareggio), la scia dell'ultimo episodio e il pallino dell'agente.
- `drawChart`: il grafico dei ritorni, con la curva grezza trasparente e la media mobile su 50 episodi. Disegnato a mano su canvas per non dipendere da librerie.
- `setupCanvas` moltiplica la risoluzione per `devicePixelRatio`: senza, su schermi retina tutto risulterebbe sfocato.
- **Editor**: scegli "Custom" come ambiente, seleziona un tipo di cella e clicca/trascina sulla griglia (eventi `pointer`, funzionano anche col touch). Il layout viene inviato al server con le reward e lo slip che imposti.
- I colori sono variabili CSS in `:root`, ridefinite in `@media (prefers-color-scheme: dark)`: la dashboard segue automaticamente il tema scuro del sistema.

---

## 8. I test (`tests/`)

Si eseguono con `pytest`. Sono organizzati per livello:

- `test_envs.py`: proprietà dell'ambiente. Ogni `P[s][a]` è una distribuzione; **la frequenza empirica di `step` coincide con `P`** (6000 campioni su FrozenLake, tolleranza 0.03): è il test che garantisce che model-based e model-free vedano lo stesso mondo; muri e bordi bloccano; il goal è assorbente; il dirupo riporta alla partenza; troncamento ≠ terminazione; lo slip si divide correttamente; i seed sono riproducibili; i layout invalidi vengono rifiutati. `@pytest.mark.parametrize` esegue lo stesso test su tutti gli ambienti registrati: aggiungi un ambiente e viene testato gratis.
- `test_value_iteration.py`: risultati **noti analiticamente**: V(start) = 3 su GridWorld, −13 su CliffWalking, ≈ 0.8235 su FrozenLake; il punto fisso soddisfa l'equazione di ottimalità di Bellman; la policy è greedy e normalizzata.
- `test_training.py`: usa un agente finto (`Recorder`) per verificare che il loop chiami gli hook **nell'ordine giusto**, che la valutazione non faccia imparare, che `EarlyStopping` fermi, che il training sia riproducibile.
- `test_dashboard.py`: con il `TestClient` di FastAPI apre davvero il WebSocket e verifica il protocollo: start → iterazioni → done → rollout, episodi in streaming, editor + modalità gioco, errori che non chiudono la connessione.
- `test_algorithms_learn.py`: **i test per i tuoi algoritmi**. Ogni algoritmo online deve risolvere GridWorld in 500 episodi; Policy Evaluation deve coincidere con la soluzione esatta del sistema lineare `(I − γP_π)V = R_π`; Policy Iteration deve coincidere con Value Iteration. Finché un algoritmo è uno stub il test viene **saltato** (`skipped`); appena lo implementi diventa un test vero. Quindi il tuo ciclo di lavoro è: scrivi l'algoritmo → `pytest` → guardalo nella dashboard.

La CI (`.github/workflows/tests.yml`) esegue `ruff check` e `pytest` su Python 3.10 e 3.12 a ogni push: il badge verde nel README è parte del "dimostrare che so scrivere codice pulito".

---

## 9. Il tuo percorso

Nell'ordine del libro, ognuno con il suo file:

1. **Bandit** (cap. 2): `algorithms/bandits/bandits.py`. ε-greedy, UCB, gradient bandit. Esercizio extra: riproduci la Figura 2.2 con il `BanditTestbed`.
2. **Policy Evaluation** (4.1): `dp/policy_iteration.py`. Suggerimento: è una riga con `self.q_from_v`. Il test ti confronta con la soluzione esatta.
3. **Policy Iteration** (4.3): stesso file. Confronta nella dashboard il numero di iterazioni con Value Iteration.
4. **Monte Carlo control** (5.4): `mc/mc_control.py`. Provalo su GridWorld; poi su CliffWalking e osserva quanto soffre per gli episodi lunghi.
5. **SARSA e Q-learning** (6.4, 6.5): `td/td_control.py`. Su CliffWalking riproduci l'Esempio 6.6: SARSA impara il percorso sicuro, Q-learning quello sul bordo. La dashboard te lo mostra con le frecce.
6. **Expected SARSA** (6.6): stesso file.
7. **n-step SARSA** (7.2): `td/n_step.py`.
8. **Dyna-Q** (8.2): `planning/dyna_q.py`. Confronta la velocità di apprendimento con Q-learning al variare di `planning_steps`.

Per ognuno: implementa → `pytest tests/test_algorithms_learn.py` → `python -m rlx.dashboard` e guardalo imparare. Dopo il tabulare, la stessa struttura (Env, Agent, loop, callback) si estende a Deep RL (ambienti con stati continui, agenti con reti neurali) e a MARL (un ambiente che riceve un'azione per agente).

### Come aggiungere un ambiente

```python
# rlx/envs/classic.py
class MyMaze(GridEnv):
    def __init__(self, seed=None):
        super().__init__(["S..#", ".#..", "...G"], GridRewards(step=-1, goal=10), seed=seed, name="MyMaze")

# rlx/envs/__init__.py, dentro ENVS
EnvSpec("mymaze", "My maze", "Descrizione.", MyMaze),
```

È già nella dashboard e già testato dai test parametrizzati.
