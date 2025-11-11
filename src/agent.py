"""
agent.py
SalesCode.ai Final Round Qualifier: LiveKit Voice Interruption Handling

This agent implements a solution for the SalesCode.ai challenge.
It extends the LiveKit Agents framework to intelligently handle
voice interruptions.

Core Logic:
1.  **State Tracking**: The agent maintains an `is_speaking` boolean flag,
    protected by an `asyncio.Lock`.
2.  **TTS Wrapping**: To accurately set the `is_speaking` flag, the agent's
    `say` method is overridden. More importantly, the TTS inference object
    itself (e.g., `inference.TTS`) is wrapped at the `entrypoint` level.
    This captures *all* audio generation, not just `agent.say()`,
    ensuring the flag is correct even during complex agent interactions.
3.  **STT Interception**: The `stt_node` method is overridden to inspect
    incoming `stt.SpeechEvent` data.
4.  **Filtering Logic**: If `is_speaking` is True, the `stt_node`
    normalizes the transcript text. If the normalized tokens consist
    *only* of words from the `ignored_words` set (e.g., "uh" or "uh umm"),
    the event's text is cleared, effectively "muting" the filler word
    for the agent's LLM.
5.  **Dynamic Configuration**: A function tool, `update_filler_words`,
    allows the filler word list to be updated live during conversation
    by asking the agent to add words to the list and then it will add them,
    addressing the optional bonus challenge.
6.  **Persistence**: The filler word list is persisted to
    `ignored_words.json` to retain settings across sessions.
"""

import asyncio
import logging
import json
import os
import re
from typing import AsyncIterable, AsyncGenerator, Any
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    inference,
    metrics,
    stt,
    ModelSettings,
    function_tool,
    RunContext,
)
from livekit.agents.utils import AudioBuffer
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit import rtc

# --- Setup ---
# Configure logging for detailed debug output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("agent")
logger.setLevel(logging.DEBUG)  # Set to DEBUG for challenge validation

load_dotenv(".env.local")

# --- Constants ---
DEFAULT_FILLER_WORDS = {"uh", "umm", "hmm", "haan"}
PERSISTENCE_FILE = "ignored_words.json"

# --- Utility Functions ---

def _normalize_and_tokens(text: str) -> list[str]:
    """
    Normalizes a transcript string for filler word matching.
    - Removes punctuation (except internal word characters)
    - Converts to lowercase
    - Strips whitespace
    - Splits into tokens
    """
    if not text:
        return []
    # remove punctuation except underscore and unicode word characters
    cleaned = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    cleaned = cleaned.strip().lower()
    if not cleaned:
        return []
    return cleaned.split()


# --- Core Agent Logic ---

class Assistant(Agent):
    """
    An Agent implementation that ignores filler words during its own speech.

    This class overrides key methods from the base `Agent` to implement
    the challenge logic without modifying core SDK code.
    """
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a helpful voice AI assistant. The user is interacting with you via voice.
Keep responses concise and friendly. Use update_filler_words tool when asked.
""",
        )

        # --- State and Synchronization ---
        # A single lock protects all state variables (is_speaking, _speech_tasks, ignored_words)
        # to ensure atomic reads/writes, critical for thread-safety.
        self._lock = asyncio.Lock()
        
        # Tracks if the agent's TTS is currently generating audio.
        self.is_speaking = False
        
        # Counter for concurrent speech tasks. `is_speaking` is True if > 0.
        self._speech_tasks = 0

        # Load persisted ignored words, falling back to defaults.
        self.ignored_words = self._load_persistent_words()
        logger.info("Assistant initialized, loaded %d filler words: %s", 
                    len(self.ignored_words), list(self.ignored_words))

    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        settings: ModelSettings,
    ) -> AsyncGenerator[stt.SpeechEvent, None]:
        """
        Overrides the STT node to intercept and filter speech events.

        This is the core of the interruption handling logic.
        It checks each STT event against the `ignored_words` list *only*
        if the agent is currently speaking.
        """
        speech_events = super().stt_node(audio, settings)

        async for event in speech_events:
            # --- 1. Passthrough non-speech or empty events ---
            if not getattr(event, "alternatives", None) or not event.alternatives:
                logger.debug("STT event has no alternatives; passing through.")
                yield event
                continue

            alt = event.alternatives[0]
            raw_text = (alt.text or "").strip()
            if not raw_text:
                logger.debug("STT alternative text empty; passing through.")
                yield event
                continue

            # Only filter INTERIM and FINAL transcripts.
            if event.type not in (stt.SpeechEventType.INTERIM_TRANSCRIPT, stt.SpeechEventType.FINAL_TRANSCRIPT):
                logger.debug("STT event type %s not subject to filler filtering; passing through.", event.type)
                yield event
                continue

            # --- 2. Normalize transcript ---
            tokens = _normalize_and_tokens(raw_text)
            logger.debug("STT Received raw: %r -> tokens: %s (event.type=%s)", raw_text, tokens, event.type)

            if not tokens:
                logger.debug("Normalization produced no tokens; passing through.")
                yield event
                continue

            # --- 3. Atomically check state ---
            # Acquire lock to get a consistent snapshot of speaking state
            # and the current set of ignored words.
            async with self._lock:
                speaking = self.is_speaking
                ignored_set = set(self.ignored_words)  # Use a copy

            # --- 4. Apply filtering logic ---
            # Check for two conditions:
            # a) The transcript is a single token (e.g., "uh") that is in the ignored set.
            # b) The transcript has multiple tokens (e.g., "uh umm") and *all*
            #    of them are in the ignored set.
            single_token = len(tokens) == 1
            single_token_matches = single_token and (tokens[0] in ignored_set)
            all_tokens_match = all(tok in ignored_set for tok in tokens)

            # Detailed logging for debugging
            logger.debug(
                "Filter decision components: speaking=%s single_token=%s single_match=%s all_tokens_match=%s ignored_set_sample=%s",
                speaking, single_token, single_token_matches, all_tokens_match, sorted(list(ignored_set))[:10],
            )

            # --- 5. Decision: Filter or Pass ---
            if speaking and (single_token_matches or all_tokens_match):
                # Scenario: Agent speaking, user says "uh" or "uh umm"
                # ACTION: Filter the event.
                logger.info("Filtered filler while speaking: raw=%r normalized=%s (single_match=%s all_match=%s)",
                            raw_text, tokens, single_token_matches, all_tokens_match)
                
                # We "filter" by clearing the text. This passes an "empty"
                # event downstream, which is safer than dropping the event
                # entirely, as it maintains the event flow.
                try:
                    event.alternatives[0].text = ""
                except Exception:
                    # Fallback: if we can't mutate the event, drop it.
                    logger.warning("Unable to clear event.alternatives[0].text; dropping event as fallback.")
                    continue
                
                setattr(event, "_filtered_filler", True) # Mark for debug
                yield event
                continue
            
            # Scenario: Agent quiet, or user says a real word 
            # (e.g., "wait"), or a mix (e.g., "umm stop").
            # ACTION: Pass the event through unmodified.
            logger.debug("Passing STT event to agent: %r", raw_text)
            yield event

    async def say(self, text: str, save_to_history: bool = True) -> AsyncIterable[AudioBuffer]:
        """
        Overrides the agent's `say` method to track speaking state.
        
        This catches speech initiated *by the agent logic* (e.g., `await agent.say(...)`).
        """
        logger.debug("Assistant.say() called")
        async with self._lock:
            if self._speech_tasks == 0:
                logger.info("Started Saying (via Assistant.say)")
                self.is_speaking = True
            self._speech_tasks += 1

        try:
            async for chunk in super().say(text, save_to_history):
                yield chunk
        finally:
            # Decrement counter and update state when speech finishes
            async with self._lock:
                self._speech_tasks -= 1
                if self._speech_tasks == 0:
                    logger.info("Stopped Saying (via Assistant.say)")
                    self.is_speaking = False

    # --- Bonus Challenge: Dynamic Configuration ---

    @function_tool
    async def update_filler_words(
        self,
        context: RunContext,
        words_to_add: list[str] | None = None,
        words_to_remove: list[str] | None = None,
    ):
        """
        An LLM tool to dynamically update the list of ignored filler words.
        
        Example user prompts:
        - "Add 'like' and 'y'know' to the filler words."
        - "Remove 'haan' from the filler list."
        """
        async with self._lock:
            updated = False
            if words_to_add:
                # Normalize words before adding
                to_add = {w.strip().lower() for w in words_to_add if w and w.strip()}
                if to_add:
                    self.ignored_words.update(to_add)
                    logger.info("Added to ignored_words: %s", to_add)
                    updated = True
            if words_to_remove:
                # Normalize words before removing
                to_remove = {w.strip().lower() for w in words_to_remove if w and w.strip()}
                if to_remove:
                    self.ignored_words.difference_update(to_remove)
                    logger.info("Removed from ignored_words: %s", to_remove)
                    updated = True

            current_words_list = sorted(list(self.ignored_words))
            logger.info("Current ignored_words: %s", current_words_list)

            if updated:
                # Persist changes to disk
                await self._save_persistent_words()

            return f"Successfully updated filler words. The new list is: {current_words_list}"

    # --- Persistence Helpers ---

    def _load_persistent_words(self) -> set:
        """
        Loads the ignored words set from PERSISTENCE_FILE.
        Creates the file with defaults if it doesn't exist.
        """
        if not os.path.exists(PERSISTENCE_FILE):
            try:
                with open(PERSISTENCE_FILE, "w", encoding="utf-8") as f:
                    json.dump(list(DEFAULT_FILLER_WORDS), f, indent=2, ensure_ascii=False)
                logger.info("Created persistence file with default filler words.")
                return DEFAULT_FILLER_WORDS.copy()
            except Exception as e:
                logger.error("Failed to create persistence file: %s. Using defaults.", e)
                return DEFAULT_FILLER_WORDS.copy()

        try:
            with open(PERSISTENCE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    logger.info("Loaded %d persistent filler words from %s", len(data), PERSISTENCE_FILE)
                    return set(data)
                else:
                    logger.warning("Persistence file format unexpected; resetting to defaults.")
                    self._write_to_file(list(DEFAULT_FILLER_WORDS)) # Fix file
                    return DEFAULT_FILLER_WORDS.copy()
        except Exception as e:
            logger.error("Failed to load persistent fillers from %s: %s. Using defaults.", PERSISTENCE_FILE, e)
            return DEFAULT_FILLER_WORDS.copy()

    def _write_to_file(self, words_list: list):
        """Synchronous file write helper (to be run in a thread)."""
        with open(PERSISTENCE_FILE, "w", encoding="utf-8") as f:
            json.dump(words_list, f, indent=2, ensure_ascii=False)

    async def _save_persistent_words(self):
        """
        Asynchronously saves the current ignored_words set to disk.
        Uses `asyncio.to_thread` to avoid blocking the event loop.
        """
        try:
            words_list = sorted(list(self.ignored_words))
            await asyncio.to_thread(self._write_to_file, words_list)
            logger.info("Persisted %d filler words to %s", len(words_list), PERSISTENCE_FILE)
        except Exception as e:
            logger.error("Failed to save persistent fillers: %s", e)


# --- Entrypoint & TTS State-Tracking Wrappers ---

def prewarm(proc: JobProcess):
    """Prewarms models before the agent starts."""
    proc.userdata["vad"] = silero.VAD.load()


class _AsyncCMProxy:
    """
    A proxy to wrap async context managers returned by TTS methods.
    
    This ensures `is_speaking` is set correctly even when the TTS
    method returns a streaming context manager instead of a direct coroutine.
    """
    def __init__(self, inner_cm: Any, assistant: Assistant, name: str):
        self._inner = inner_cm
        self._assistant = assistant
        self._name = name

    async def __aenter__(self):
        async with self._assistant._lock:
            if self._assistant._speech_tasks == 0:
                logger.info("TTS-wrapper: Started Saying (via tts_infer.%s CM)", self._name)
                self._assistant.is_speaking = True
            self._assistant._speech_tasks += 1
        return await self._inner.__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        try:
            return await self._inner.__aexit__(exc_type, exc, tb)
        finally:
            async with self._assistant._lock:
                self._assistant._speech_tasks -= 1
                if self._assistant._speech_tasks == 0:
                    logger.info("TTS-wrapper: Stopped Saying (via tts_infer.%s CM)", self._name)
                    self._assistant.is_speaking = False


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the agent worker.
    Sets up the agent, session, and state-tracking wrappers.
    """
    ctx.log_context_fields = {"room": ctx.room.name}

    assistant = Assistant()

    # --- TTS Inference Wrapping ---
    # This is critical for robustly tracking the `is_speaking` state.
    # We wrap the TTS object *before* it's given to the AgentSession.
    # This captures all speech, including preemptive generation,
    # which may not call `Assistant.say()`.
    
    tts_infer = inference.TTS(
        model="cartesia/sonic-3",
        voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc", # Example voice
    )

    # List of potential methods on a TTS object that could produce audio
    tts_candidates = [
        "__call__", "synthesize", "generate", "run", "produce", "stream",
        "stream_generate", "synthesize_stream", "stream_synthesize",
    ]

    for cand in tts_candidates:
        if not hasattr(tts_infer, cand):
            continue

        orig_callable = getattr(tts_infer, cand)

        # Case 1: The original method is an async function
        if asyncio.iscoroutinefunction(orig_callable):
            def make_async_wrapper(orig, name):
                async def wrapper(*a, **kw):
                    async with assistant._lock:
                        if assistant._speech_tasks == 0:
                            logger.info("TTS-wrapper: Started Saying (via tts_infer.%s)", name)
                            assistant.is_speaking = True
                        assistant._speech_tasks += 1
                    try:
                        return await orig(*a, **kw)
                    finally:
                        async with assistant._lock:
                            assistant._speech_tasks -= 1
                            if assistant._speech_tasks == 0:
                                logger.info("TTS-wrapper: Stopped Saying (via tts_infer.%s)", name)
                                assistant.is_speaking = False
                return wrapper
            wrapped = make_async_wrapper(orig_callable, cand)
            setattr(tts_infer, cand, wrapped)
            logger.debug("Installed async wrapper for tts_infer.%s", cand)
            continue

        # Case 2: The original method is a sync function
        # (which might return an async context manager)
        def make_sync_wrapper(orig_sync, name_sync):
            def wrapper(*a, **kw):
                inner = orig_sync(*a, **kw)
                
                # If the sync method returns an async context manager,
                # return our proxy to wrap its __aenter__ and __aexit__.
                if hasattr(inner, "__aenter__") and hasattr(inner, "__aexit__"):
                    return _AsyncCMProxy(inner, assistant, name_sync)
                else:
                    # If it returns a simple value, wrap it in a
                    # coroutine that manages state.
                    async def run_and_return():
                        async with assistant._lock:
                            if assistant._speech_tasks == 0:
                                logger.info("TTS-wrapper: Started Saying (via tts_infer.%s sync-return)", name_sync)
                                assistant.is_speaking = True
                            assistant._speech_tasks += 1
                        try:
                            return inner
                        finally:
                            async with assistant._lock:
                                assistant._speech_tasks -= 1
                                if assistant._speech_tasks == 0:
                                    logger.info("TTS-wrapper: Stopped Saying (via tts_infer.%s sync-return)", name_sync)
                                    assistant.is_speaking = False
                    return run_and_return()
            return wrapper

        wrapped = make_sync_wrapper(orig_callable, cand)
        setattr(tts_infer, cand, wrapped)
        logger.debug("Installed sync wrapper for tts_infer.%s", cand)

    # --- AgentSession Setup ---
    session = AgentSession(
        false_interruption_timeout=0.2, # Low timeout for responsiveness
        stt=inference.STT(model="assemblyai/universal-streaming", language="en", extra_kwargs={"disfluencies": True}),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=tts_infer, # Pass in our *wrapped* TTS object
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("Usage: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    logger.info("Starting session with Assistant instance id=%s", id(assistant))

    # Start the session
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))