"""Browser-provided dictation controls for the AI Analyst question box."""

import streamlit as st


_HTML = """
<div class="dictation">
  <div class="controls">
    <label for="dictation-language">Language</label>
    <select id="dictation-language">
      <option value="en-GB">English (UK)</option>
      <option value="en-US">English (US)</option>
      <option value="pl-PL">Polski</option>
      <option value="de-DE">Deutsch</option>
      <option value="fr-FR">Français</option>
    </select>
    <button id="dictation-start" type="button">Start dictation (online)</button>
    <button id="dictation-stop" type="button" disabled>Stop</button>
    <button id="dictation-reset" type="button">Reset dictation</button>
  </div>
  <div id="dictation-status" role="status" aria-live="polite">
    Browser dictation processes audio online. It does not send audio to OpenRouter.
  </div>
</div>
"""

_CSS = """
.dictation { margin: 0.25rem 0 0.75rem; font: inherit; color: var(--st-text-color); }
.controls { display: flex; align-items: center; flex-wrap: wrap; gap: 0.5rem; }
.controls label { font-size: 0.9rem; }
.controls select, .controls button {
  min-height: 2.35rem; border: 1px solid var(--st-border-color);
  border-radius: 0.5rem; padding: 0.4rem 0.75rem;
  color: var(--st-text-color); background: var(--st-background-color);
  font: inherit;
}
.controls button { cursor: pointer; }
.controls button:disabled { cursor: not-allowed; opacity: 0.55; }
#dictation-reset { margin-left: auto; }
#dictation-status { margin-top: 0.4rem; color: var(--st-secondary-text-color); font-size: 0.85rem; }
"""

_JS = r"""
export default function(component) {
  const { parentElement, setStateValue } = component;
  if (parentElement.__portfolioDictation) return;

  const language = parentElement.querySelector("#dictation-language");
  const startButton = parentElement.querySelector("#dictation-start");
  const stopButton = parentElement.querySelector("#dictation-stop");
  const resetButton = parentElement.querySelector("#dictation-reset");
  const status = parentElement.querySelector("#dictation-status");
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const controller = {
    recognition: null,
    wantsListening: false,
    retryTimer: null,
    flushTimer: null,
    pendingText: "",
    generation: 0,
    disposed: false,
  };
  parentElement.__portfolioDictation = controller;

  const setStatus = (message) => { status.textContent = message; };

  // Use the browser's preferred language by default, while leaving a simple
  // override for people speaking a different language.
  const preferredLanguages = navigator.languages?.length
    ? navigator.languages : [navigator.language || "en-US"];
  for (const preferred of preferredLanguages) {
    const options = Array.from(language.options);
    const match = options.find((option) => option.value.toLowerCase() === preferred.toLowerCase())
      || options.find((option) => option.value.slice(0, 2).toLowerCase() === preferred.slice(0, 2).toLowerCase());
    if (match) {
      language.value = match.value;
      break;
    }
  }

  if (!Recognition) {
    startButton.disabled = true;
    setStatus("Dictation is not supported by this browser. You can type your question or use your device's voice typing in the question box.");
  } else {
    setStatus("Browser dictation sends audio to the browser's online speech service. Audio is not sent to OpenRouter.");
  }

  const flushTranscript = () => {
    clearTimeout(controller.flushTimer);
    const text = controller.pendingText.trim();
    controller.pendingText = "";
    if (text && !controller.disposed) {
      setStateValue("transcript_event", {
        id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
        text,
      });
    }
  };

  const stopListening = (manual = true) => {
    controller.wantsListening = false;
    clearTimeout(controller.retryTimer);
    flushTranscript();
    startButton.disabled = !Recognition;
    stopButton.disabled = true;
    if (controller.recognition) {
      try { controller.recognition.stop(); } catch (_) { /* already stopped */ }
    }
    if (manual) setStatus("Dictation stopped. Recognized text has been added to the question box.");
  };

  const resetDictation = () => {
    // Change generation before aborting so late recognition callbacks cannot
    // append a phrase after the user has reset the question.
    controller.generation += 1;
    controller.wantsListening = false;
    clearTimeout(controller.retryTimer);
    clearTimeout(controller.flushTimer);
    controller.pendingText = "";
    if (controller.recognition) {
      try { controller.recognition.abort(); } catch (_) { /* already stopped */ }
    }
    controller.recognition = null;
    startButton.disabled = !Recognition;
    stopButton.disabled = true;
    setStateValue("reset_event", { id: `${Date.now()}-${Math.random().toString(36).slice(2)}` });
    setStatus("Dictation reset. The question box has been cleared.");
  };

  const startListening = () => {
    if (!Recognition || controller.wantsListening) return;
    const generation = ++controller.generation;
    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.lang = language.value;
    controller.recognition = recognition;
    controller.wantsListening = true;
    startButton.disabled = true;
    stopButton.disabled = false;
    setStatus("Listening with browser dictation. Completed phrases are added to the question as you speak.");

    recognition.onresult = (event) => {
      if (generation !== controller.generation) return;
      let finalText = "";
      let interimText = "";
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const text = result[0]?.transcript || "";
        if (result.isFinal) finalText += text;
        else interimText += text;
      }
      if (finalText.trim()) {
        controller.pendingText += `${controller.pendingText ? " " : ""}${finalText.trim()}`;
        clearTimeout(controller.flushTimer);
        // Batch adjacent recognition fragments to reduce Streamlit reruns
        // while a sentence is still being recognized.
        controller.flushTimer = setTimeout(flushTranscript, 500);
      }
      if (controller.wantsListening) {
        setStatus(interimText.trim()
          ? `Listening… ${interimText.trim()}`
          : "Listening with browser dictation. Completed phrases are added as you speak.");
      }
    };

    recognition.onerror = (event) => {
      if (generation !== controller.generation) return;
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        controller.wantsListening = false;
        startButton.disabled = false;
        stopButton.disabled = true;
        setStatus("Microphone access was blocked. Allow microphone access for this site, then try again.");
      } else if (event.error === "language-not-supported") {
        controller.wantsListening = false;
        startButton.disabled = false;
        stopButton.disabled = true;
        setStatus(`Browser dictation does not support ${language.value}. Choose another language or type your question.`);
      } else if (event.error !== "no-speech" && controller.wantsListening) {
        setStatus(`Browser dictation paused (${event.error}). It will try to reconnect.`);
      }
    };

    recognition.onend = () => {
      if (!controller.wantsListening || controller.disposed || generation !== controller.generation) return;
      setStatus("Browser dictation paused briefly. Reconnecting…");
      controller.retryTimer = setTimeout(() => {
        if (!controller.wantsListening || controller.disposed || generation !== controller.generation) return;
        try { recognition.start(); }
        catch (_) {
          controller.retryTimer = setTimeout(() => {
            if (!controller.wantsListening || controller.disposed || generation !== controller.generation) return;
            try { recognition.start(); }
            catch (_) { setStatus("Browser dictation stopped. Press Start dictation to continue; completed phrases remain in the question box."); }
          }, 800);
        }
      }, 350);
    };

    try { recognition.start(); }
    catch (error) {
      controller.wantsListening = false;
      startButton.disabled = false;
      stopButton.disabled = true;
      setStatus(`Browser dictation could not start (${error?.message || "browser error"}). Check microphone access and try again.`);
    }
  };

  startButton.addEventListener("click", startListening);
  stopButton.addEventListener("click", () => stopListening(true));
  resetButton.addEventListener("click", resetDictation);

  return () => {
    flushTranscript();
    controller.disposed = true;
    controller.wantsListening = false;
    clearTimeout(controller.retryTimer);
    clearTimeout(controller.flushTimer);
    if (controller.recognition) {
      try { controller.recognition.abort(); } catch (_) { /* already stopped */ }
    }
    startButton.removeEventListener("click", startListening);
    resetButton.removeEventListener("click", resetDictation);
  };
}
"""


@st.cache_resource
def _portfolio_question_dictation_component():
    return st.components.v2.component(
        "portfolio_question_browser_dictation",
        html=_HTML,
        css=_CSS,
        js=_JS,
    )


def portfolio_question_input_component(**kwargs):
    """Mount browser dictation controls and return their state."""
    return _portfolio_question_dictation_component()(**kwargs)
