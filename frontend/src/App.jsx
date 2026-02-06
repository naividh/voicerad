import React, { useState, useRef } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "";

export default function App() {
  const [sid, setSid] = useState(null);
  const [interp, setInterp] = useState(null);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);
  const [imgFile, setImgFile] = useState(null);
  const [transcript, setTranscript] = useState("");
  const [recording, setRecording] = useState(false);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);

  React.useEffect(() => {
    const on = () => setIsOnline(true);
    const off = () => setIsOnline(false);
    window.addEventListener("online", on);
    window.addEventListener("offline", off);
    return () => { window.removeEventListener("online", on); window.removeEventListener("offline", off); };
  }, []);

  const pickImage = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setImgFile(f);
    const r = new FileReader();
    r.onload = (ev) => setPreview(ev.target.result);
    r.readAsDataURL(f);
  };

  const startRec = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mr = new MediaRecorder(stream);
      recorderRef.current = mr;
      const chunks = [];
      mr.ondataavailable = (e) => chunks.push(e.data);
      mr.onstop = async () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        const fd = new FormData();
        fd.append("audio", blob);
        try {
          const res = await fetch(API + "/api/voice/transcribe", { method: "POST", body: fd });
          const d = await res.json();
          setTranscript(d.transcript);
        } catch { setTranscript("(transcription unavailable)"); }
      };
      mr.start();
      setRecording(true);
    } catch { alert("Microphone access denied"); }
  };

  const stopRec = () => {
    recorderRef.current?.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setRecording(false);
  };

  const submit = async () => {
    if (!imgFile || !transcript) return alert("Upload image and ask a question first");
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("image", imgFile);
      fd.append("question", transcript);
      const res = await fetch(API + "/api/interpret/start-session", { method: "POST", body: fd });
      const d = await res.json();
      if (d.session_id) { setSid(d.session_id); setInterp(d.interpretation); }
    } catch { alert("Connection error"); }
    setLoading(false);
  };

  const finalize = async () => {
    if (!sid) return;
    setLoading(true);
    try {
      const res = await fetch(API + "/api/interpret/finalize/" + sid, { method: "POST" });
      const d = await res.json();
      setInterp(d.final_report);
      setSid(null);
    } catch { alert("Error finalizing"); }
    setLoading(false);
  };

  const speak = (text) => {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  };

  const reset = () => { setSid(null); setInterp(null); setPreview(null); setImgFile(null); setTranscript(""); };

  return (
    <div className="vr">
      <header className="hd">
        <h1>VoiceRad</h1>
        <p>Voice-Controlled Radiology Assistant</p>
        <span className={"badge " + (isOnline ? "on" : "off")}>{isOnline ? "Online" : "Offline"}</span>
      </header>

      <main className="mn">
        {!sid ? (
          <>
            <div className="upload">
              <input type="file" id="img" accept="image/*" onChange={pickImage} hidden />
              <label htmlFor="img" className="uplbl">
                <span className="icon">📸</span>
                <span>Upload medical image</span>
              </label>
              {preview && <img src={preview} alt="preview" className="prev" />}
            </div>

            <div className="voice">
              {!recording
                ? <button onClick={startRec} className="btn rec">🎤 Record Question</button>
                : <button onClick={stopRec} className="btn stp">⏹ Stop</button>}
              {transcript && <p className="trans"><b>Q:</b> {transcript}</p>}
              <button onClick={submit} disabled={!imgFile || !transcript || loading} className="btn sub">
                {loading ? "Analysing..." : "Analyse Image"}
              </button>
            </div>
          </>
        ) : (
          <div className="results">
            <h2>Interpretation</h2>
            <pre className="report">{interp}</pre>
            <div className="actions">
              <button onClick={() => speak(interp)} className="btn spk">🔊 Listen</button>
              <button onClick={finalize} disabled={loading} className="btn fin">📋 Final Report</button>
              <button onClick={reset} className="btn rst">🔄 New</button>
            </div>
          </div>
        )}

        {loading && <div className="ld"><div className="spinner" /><p>Processing...</p></div>}
      </main>

      <footer className="ft">
        <p>⚠️ VoiceRad is a demo. Always verify with medical professionals.</p>
        <p>Powered by Google MedGemma &amp; MedASR | Kaggle MedGemma Impact Challenge 2026</p>
      </footer>
    </div>
  );
}
