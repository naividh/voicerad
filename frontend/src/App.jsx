import React, { useState, useRef, useCallback } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "";

// ── Custom Hook: Voice Recorder ─────────────────────────
function useVoiceRecorder(onTranscript) {
  const [recording, setRecording] = useState(false);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      const chunks = [];
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        const form = new FormData();
        form.append("audio", blob);
        try {
          const res = await fetch(API + "/api/voice/transcribe", { method: "POST", body: form });
          const data = await res.json();
          onTranscript(data.transcript);
        } catch {
          onTranscript("(transcription unavailable)");
        }
      };
      recorder.start();
      setRecording(true);
    } catch {
      alert("Microphone access denied");
    }
  }, [onTranscript]);

  const stop = useCallback(() => {
    recorderRef.current?.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setRecording(false);
  }, []);

  return { recording, start, stop };
}

// ── Component: Image Upload ─────────────────────────────
function ImageUpload({ preview, onImageSelected }) {
  const handleChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => onImageSelected(file, ev.target.result);
    reader.readAsDataURL(file);
  };

  return (
    <div className="upload">
      <input type="file" id="img" accept="image/*" onChange={handleChange} hidden />
      <label htmlFor="img" className="uplbl">
        <span className="icon">📸</span>
        <span>Upload medical image</span>
      </label>
      {preview && <img src={preview} alt="Medical image preview" className="prev" />}
    </div>
  );
}

// ── Component: Interpretation Results ───────────────────
function InterpretationView({ interpretation, sessionActive, loading, onListen, onFinalize, onReset }) {
  return (
    <div className="results">
      <h2>{sessionActive ? "Interpretation" : "Final Report"}</h2>
      <pre className="report">{interpretation}</pre>
      <div className="actions">
        <button onClick={onListen} className="btn spk">🔊 Listen</button>
        {sessionActive && (
          <button onClick={onFinalize} disabled={loading} className="btn fin">
            {loading ? "Generating..." : "📋 Final Report"}
          </button>
        )}
        <button onClick={onReset} className="btn rst">🔄 New</button>
      </div>
    </div>
  );
}

// ── Main App ────────────────────────────────────────────
export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [interpretation, setInterpretation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [transcript, setTranscript] = useState("");
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [showResults, setShowResults] = useState(false);

  const { recording, start: startRecording, stop: stopRecording } = useVoiceRecorder(setTranscript);

  React.useEffect(() => {
    const goOnline = () => setIsOnline(true);
    const goOffline = () => setIsOnline(false);
    window.addEventListener("online", goOnline);
    window.addEventListener("offline", goOffline);
    return () => {
      window.removeEventListener("online", goOnline);
      window.removeEventListener("offline", goOffline);
    };
  }, []);

  const handleImageSelected = (file, dataUrl) => {
    setImageFile(file);
    setPreview(dataUrl);
  };

  const handleSubmit = async () => {
    if (!imageFile || !transcript) {
      return alert("Upload an image and record a question first");
    }
    setLoading(true);
    try {
      const form = new FormData();
      form.append("image", imageFile);
      form.append("question", transcript);
      const res = await fetch(API + "/api/interpret/start-session", { method: "POST", body: form });
      const data = await res.json();
      if (data.session_id) {
        setSessionId(data.session_id);
        setInterpretation(data.interpretation);
        setShowResults(true);
      }
    } catch {
      alert("Connection error — check if the backend is running");
    }
    setLoading(false);
  };

  const handleFinalize = async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const res = await fetch(API + "/api/interpret/finalize/" + sessionId, { method: "POST" });
      const data = await res.json();
      setInterpretation(data.final_report);
      setSessionId(null);
    } catch {
      alert("Error generating final report");
    }
    setLoading(false);
  };

  const handleSpeak = (text) => {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  };

  const handleReset = () => {
    setSessionId(null);
    setInterpretation(null);
    setPreview(null);
    setImageFile(null);
    setTranscript("");
    setShowResults(false);
  };

  return (
    <div className="vr">
      <header className="hd">
        <h1>VoiceRad</h1>
        <p>Voice-Controlled Radiology Assistant</p>
        <span className={"badge " + (isOnline ? "on" : "off")}>
          {isOnline ? "Online" : "Offline"}
        </span>
      </header>

      <main className="mn">
        {!showResults ? (
          <>
            <ImageUpload preview={preview} onImageSelected={handleImageSelected} />

            <div className="voice">
              {!recording ? (
                <button onClick={startRecording} className="btn rec">🎤 Record Question</button>
              ) : (
                <button onClick={stopRecording} className="btn stp">⏹ Stop Recording</button>
              )}
              {transcript && (
                <p className="trans"><b>Q:</b> {transcript}</p>
              )}
              <button
                onClick={handleSubmit}
                disabled={!imageFile || !transcript || loading}
                className="btn sub"
              >
                {loading ? "Analysing..." : "Analyse Image"}
              </button>
            </div>
          </>
        ) : (
          <InterpretationView
            interpretation={interpretation}
            sessionActive={sessionId !== null}
            loading={loading}
            onListen={() => handleSpeak(interpretation)}
            onFinalize={handleFinalize}
            onReset={handleReset}
          />
        )}

        {loading && (
          <div className="ld">
            <div className="spinner" />
            <p>Processing...</p>
          </div>
        )}
      </main>

      <footer className="ft">
        <p>⚠️ VoiceRad is a demo. Always verify with medical professionals.</p>
        <p>Powered by Google MedGemma &amp; MedASR | Kaggle MedGemma Impact Challenge 2026</p>
      </footer>
    </div>
  );
}
