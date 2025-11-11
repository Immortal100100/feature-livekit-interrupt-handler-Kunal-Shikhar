# SalesCode.ai Final Round: LiveKit Interrupt Handler

This repository contains the solution for the "LiveKit Voice Interruption Handling Challenge".

The goal is to enhance a LiveKit agent to intelligently ignore filler words (like "uh", "umm") when the agent is speaking.

## What Changed

The solution is implemented entirely within `src/agent.py` as an extension layer, with no modifications to the core LiveKit SDK.

1.  **`Assistant(Agent)` Class**: A custom agent class that overrides key methods.
2.  **`is_speaking` State**: The `Assistant` class maintains an `is_speaking` boolean flag, protected by an `asyncio.Lock` for thread-safe access.
3.  **Robust State Tracking**: The `is_speaking` flag is controlled by:
    * Overriding the `Assistant.say` method.
    * **Monkey-patching the `inference.TTS` object**: In the `entrypoint` function, the TTS object is wrapped *before* being passed to the `AgentSession`. This is the key to robustly tracking *all* speech generation, including preemptive TTS, which `agent.say` alone would miss.
4.  **`stt_node` Override**: This method intercepts all `stt.SpeechEvent` data.
    * It normalizes the incoming transcript (lowercase, remove punctuation).
    * If `is_speaking` is `True`, it checks if the normalized tokens consist *only* of words in the `ignored_words` set.
    * If a match is found, the event's text is set to `""`, effectively "muting" the filler word for the LLM while maintaining the event stream.
5.  **Configurable & Persistent Word List**:
    * The list of filler words is loaded from `ignored_words.json` on startup.
    * A default list (`uh`, `umm`, `hmm`, `haan`) is used if the file is missing.
6.  **Bonus: Dynamic Updates**:
    * An LLM function tool, `update_filler_words`, is provided.
    * This allows a user to ask the agent to add or remove words from the filler list (e.g., "add 'like' to the filler words").
    * Changes are immediately active and persisted to `ignored_words.json`.

## What Works

The solution successfully meets the core objectives and bonus challenges:

* **✅ Ignore Fillers When Agent Speaks**: Saying "uh" or "umm" while the agent is talking is ignored, and the agent continues speaking.
* **✅ Handle Multi-Word Fillers**: Handles cases like "uh umm" or "haan haan" by checking if *all* tokens in the transcript are ignored words.
* **✅ Handle Real Interruptions**: Saying "wait" or "stop" while the agent is speaking immediately stops the agent's TTS.
* **✅ Handle Mixed Interruptions**: Saying "umm okay stop" correctly registers as a real interruption (because "okay" and "stop" are not in the ignored list) and stops the agent.
* **✅ Configurable**: The filler list is easily configurable via the `ignored_words.json` file.
* **✅ (Bonus) Dynamic Updates**: The filler list can be updated in real-time via conversation using the `update_filler_words` tool.

## Known Issues

* **Fragile TTS Wrapping**: The `is_speaking` state relies on monkey-patching the `inference.TTS` object in the `entrypoint`. This wrapper code iterates over a hardcoded list of possible method names (e.g., `__call__`, `synthesize`, `stream`). If the `livekit-agents` SDK changes these internal method names, the state tracking will break.
* **VAD-Only Triggers**: The agent can be sensitive to non-speech sounds (like coughs, mic bumps, or background noises) that trigger Voice Activity Detection (VAD) but do not result in a valid STT transcript. This can cause a brief, unnecessary pause in the agent's speech. This is partially mitigated by setting `false_interruption_timeout=0.2` in the `AgentSession`, which allows the agent to resume speaking quickly if no transcript follows the VAD signal.

## Steps to Test

1.  **Clone Repo**:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create Branch** (as per instructions):
    ```bash
    git checkout -b feature/livekit-interrupt-handler-<yourname>
    ```

3.  **Setup Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download AI/ML Models**:
    This command is required by the dependencies to download VAD and other models.
    ```bash
    python src/agent.py download-files
    ```

6.  **Configure Environment**:
    * Create a `.env.local` file by copying `.env.example`.
    * Fill in your API keys for LiveKit.
    ```bash
    cp .env.example .env.local
    nano .env.local  # Or use your preferred editor
    ```
    * An `ignored_words.json` file will be created automatically on the first run with default values.

7.  **Run the Agent**:
    ```bash
    python src/agent.py
    ```

8.  **Test Scenarios**:
    * **Test 1 (Filter Filler)**: While the agent is speaking (e.g., "Hello, how can I help you?"), say "**umm**" or "**uh**".
        * *Expected*: The agent ignores you and continues speaking. Check the logs for "Filtered filler while speaking".
    * **Test 2 (Real Interruption)**: While the agent is speaking, say "**wait one second**".
        * *Expected*: The agent immediately stops speaking.
    * **Test 3 (Mixed Interruption)**: While the agent is speaking, say "**umm okay stop**".
        * *Expected*: The agent immediately stops, as the transcript contains valid words ("okay", "stop").
    * **Test 4 (Dynamic Update)**: Ask the agent, "**Can you add 'like' to the filler words list?**"
        * *Expected*: The agent should confirm. "Successfully updated... ['haan', 'hmm', 'like', 'uh', 'umm']".
    * **Test 5 (Verify Update)**: After Test 5, try to interrupt the agent by saying "**like**".
        * *Expected*: The agent should now *ignore* "like" and continue speaking.

## Environment Details

* **Python**: Python 3.10+
* **Core Dependencies**: See `requirements.txt`
* **Configuration**:
    * `.env.local`: Used for API keys (see `.env.example`).
    * `ignored_words.json`: Persists the set of filler words. Can be edited manually or updated via the agent tool.