import React, { useState, useRef, useCallback, useEffect } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "";

// -- IndexedDB helper for offline queue ---------------------
const DB_NAME = "VoiceRadDB";
const DB_VERSION = 1;
const STORE = "pending";

function openDB() {
    return new Promise((resolve, reject) => {
          const req = indexedDB.open(DB_NAME, DB_VERSION);
          req.onupgradeneeded = (e) => {
                  e.target.result.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
          };
          req.onsuccess = () => resolve(req.result);
          req.onerror = () => reject(req.error);
    });
}

async function queueOffline(formData) {
    const db = await openDB();
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).add({ formData, ts: Date.now() });
    if ("serviceWorker" in navigator && "SyncManager" in window) {
          const reg = await navigator.serviceWorker.ready;
          await reg.sync.register("sync-interpretations");
    }
}

// -- Custom Hook: Voice Recorder ----------------------------
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
                                      if (!res.ok) throw new Error(`HTTP ${res.status}`);
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

// -- Component: Image Upload --------------------------------
function ImageUpload({ preview, onImageSelected }) {
    const [dragOver, setDragOver] = useState(false);

  const processFile = (file) => {
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => onImageSelected(file, ev.target.result);
        reader.readAsDataURL(file);
  };

  const handleChange = (e) => processFile(e.target.files[0]);

  const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        processFile(e.dataTransfer.files[0]);
  };

  return (
        <div className="upload">
              <div
                        className={`uplbl ${dragOver ? "drag-over" : ""}`}
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                        onClick={() => document.getElementById("img").click()}
                      >
                      <input type="file" id="img" accept="image/*,.dcm,.dicom" onChange={handleChange} hidden />
                      <span className="icon">📸</span>
                      <span>Upload medical image</span>
                      <span className="hint">PNG, JPG, or DICOM (.dcm)</span>
              </div>
          {preview && <img src={preview} alt="Medical image preview" className="prev" />}
        </div>
      );
}

// -- Component: Conversation Turn ---------------------------
function ConversationTurn({ turn, index }) {
    return (
          <div className="turn">
                <div className="turn-header">Turn {index + 1}</div>
                <div className="turn-input"><b>You:</b> {turn.input}</div>
            {turn.response && <pre className="turn-response">{turn.response}</pre>}
          </div>
        );
}

// -- Component: Interpretation Results ----------------------
function InterpretationView({
    interpretation,
    turns,
    sessionActive,
    loading,
    onListen,
    onFinalize,
    onReset,
    onContinue,
    continueText,
    setContinueText,
    recording,
    startRecording,
    stopRecording,
}) {
    return (
          <div className="results">
                <h2>{sessionActive ? "Interpretation" : "Final Report"}</h2>
                <pre className="report">{interpretation}</pre>
          
            {turns.length > 0 && (
                    <div className="turns-list">
                              <h3>Conversation History</h3>
                      {turns.map((t, i) => <ConversationTurn key={i} turn={t} index={i} />)}
                    </div>
                )}
          
                <div className="actions">
                        <button onClick={onListen} className="btn spk">🔊 Listen</button>
                  {sessionActive && (
                      <button onClick={onFinalize} disabled={loading} className="btn fin">
                        {loading ? "Generating..." : "📋 Final Report"}
                      </button>
                        )}
                        <button onClick={onReset} className="btn rst">🔄 New</button>
                </div>
          
            {sessionActive && (
                    <div className="continue-section">
                              <h3>Refine Interpretation</h3>
                              <div className="continue-row">
                                          <input
                                                          type="text"
                                                          className="continue-input"
                                                          placeholder="Add clinical context..."
                                                          value={continueText}
                                                          onChange={(e) => setContinueText(e.target.value)}
                                                          onKeyDown={(e) => e.key === "Enter" && onContinue()}
                                                        />
                                {!recording ? (
                                    <button onClick={startRecording} className="btn mic-sm" title="Record voice">🎤</button>
                                  ) : (
                                    <button onClick={stopRecording} className="btn stp-sm" title="Stop recording">⏹</button>
                                          )}
                                          <button
                                                          onClick={onContinue}
                                                          disabled={!continueText.trim() || loading}
                                                          className="btn send-sm"
                                                        >
                                            {loading ? "..." : "➤"}
                                          </button>
                              </div>
                    </div>
                )}
          </div>
        );
}

// -- Main App -----------------------------------------------
export default function App() {
    const [sessionId, setSessionId] = useState(null);
    const [interpretation, setInterpretation] = useState(null);
    const [loading, setLoading] = useState(false);
    const [preview, setPreview] = useState(null);
    const [imageFile, setImageFile] = useState(null);
    const [transcript, setTranscript] = useState("");
    const [isOnline, setIsOnline] = useState(navigator.onLine);
    const [showResults, setShowResults] = useState(false);
    const [turns, setTurns] = useState([]);
    const [continueText, setContinueText] = useState("");
    const [error, setError] = useState(null);
  
    const { recording, start: startRecording, stop: stopRecording } = useVoiceRecorder(
          (text) => {
                  if (showResults) {
                            setContinueText(text);
                  } else {
                            setTranscript(text);
                  }
          }
        );
  
    useEffect(() => {
          const goOnline = () => setIsOnline(true);
          const goOffline = () => setIsOnline(false);
          window.addEventListener("online", goOnline);
          window.addEventListener("offline", goOffline);
          return () => {
                  window.removeEventListener("online", goOnline);
                  window.removeEventListener("offline", goOffline);
          };
    }, []);
  
    // Clear error after 5s
    useEffect(() => {
          if (error) {
                  const t = setTimeout(() => setError(null), 5000);
                  return () => clearTimeout(t);
          }
    }, [error]);
  
    const handleImageSelected = (file, dataUrl) => {
          setImageFile(file);
          setPreview(dataUrl);
          setError(null);
    };
  
    const handleSubmit = async () => {
          if (!imageFile || !transcript) {
                  return setError("Upload an image and record a question first");
          }
          setLoading(true);
          setError(null);
          try {
                  const form = new FormData();
                  form.append("image", imageFile);
                  form.append("question", transcript);
            
                  if (!isOnline) {
                            await queueOffline(form);
                            setError("Offline - request queued for sync");
                            setLoading(false);
                            return;
                  }
            
                  const res = await fetch(API + "/api/interpret/start-session", { method: "POST", body: form });
                  if (!res.ok) {
                            const err = await res.json().catch(() => ({}));
                            throw new Error(err.detail || `Server error ${res.status}`);
                  }
                  const data = await res.json();
                  if (data.session_id) {
                            setSessionId(data.session_id);
                            setInterpretation(data.interpretation);
                            setShowResults(true);
                            setTurns([]);
                  }
          } catch (e) {
                  setError(e.message || "Connection error - check if backend is running");
          }
          setLoading(false);
    };
  
    const handleContinue = async () => {
          if (!sessionId || !continueText.trim()) return;
          setLoading(true);
          setError(null);
          try {
                  const form = new FormData();
                  form.append("answer", continueText.trim());
                  const res = await fetch(API + "/api/interpret/continue/" + sessionId, {
                            method: "POST",
                            body: form,
                  });
                  if (!res.ok) throw new Error(`Server error ${res.status}`);
                  const data = await res.json();
                  setTurns((prev) => [...prev, { input: continueText.trim(), response: data.refined_interpretation }]);
                  setInterpretation(data.refined_interpretation);
                  setContinueText("");
          } catch (e) {
                  setError(e.message || "Error refining interpretation");
          }
          setLoading(false);
    };
  
    const handleFinalize = async () => {
          if (!sessionId) return;
          setLoading(true);
          setError(null);
          try {
                  const res = await fetch(API + "/api/interpret/finalize/" + sessionId, { method: "POST" });
                  if (!res.ok) throw new Error(`Server error ${res.status}`);
                  const data = await res.json();
                  setInterpretation(data.final_report);
                  setSessionId(null);
          } catch (e) {
                  setError(e.message || "Error generating final report");
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
          setTurns([]);
          setContinueText("");
          setError(null);
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
          
            {error && <div className="error-banner">{error}</div>}
          
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
                                    {transcript && <p className="trans"><b>Q:</b> {transcript}</p>}
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
                                    turns={turns}
                                    sessionActive={sessionId !== null}
                                    loading={loading}
                                    onListen={() => handleSpeak(interpretation)}
                                    onFinalize={handleFinalize}
                                    onReset={handleReset}
                                    onContinue={handleContinue}
                                    continueText={continueText}
                                    setContinueText={setContinueText}
                                    recording={recording}
                                    startRecording={startRecording}
                                    stopRecording={stopRecording}
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
                        <p>Powered by Google MedGemma & MedASR | Kaggle MedGemma Impact Challenge 2026</p>
                </footer>
          </div>
        );
}</></div>
