# VoiceRad — 3-Minute Video Demo Script

> **Target length:** 3:00  
> **Format:** Screen recording with voice-over narration  
> **Tools:** OBS / Loom / any screen recorder + microphone  

---

## SCENE 1 — Hook (0:00 – 0:20)

**[Screen: Title card — "VoiceRad" logo, tagline]**

> **Narration:**  
> "4.5 billion people lack access to radiology services.  
> 42,000 radiologist positions will be unfilled by 2033.  
> In rural clinics, a single doctor handles everything — and they don't have time to type.  
> Meet VoiceRad."

**[Transition: fade to phone mockup / laptop showing app]**

---

## SCENE 2 — Live Demo Walk-Through (0:20 – 1:40)

### 2a — Open the App (0:20 – 0:30)

**[Screen: Browser at localhost:3000 showing VoiceRad UI]**

> "VoiceRad is a Progressive Web App. Open it on any phone or laptop — no install required.  
> The interface is built for one-handed use in a busy clinic."

### 2b — Voice Input (0:30 – 0:50)

**[Screen: Tap "Record Question" → speak → transcript appears]**

> "The clinician taps Record and asks their question naturally:  
> 'What do you see in this chest X-ray?'  
> VoiceRad transcribes it in real time using medical speech recognition."

### 2c — Image Upload (0:50 – 1:00)

**[Screen: Tap upload area → select chest X-ray image → upload confirmation]**

> "Next, they upload the image — snap it with the phone camera,  
> or pick an existing DICOM or JPEG. The image is sent to the backend."

### 2d — AI Interpretation (1:00 – 1:20)

**[Screen: Click "Analyse Image" → interpretation card appears with findings]**

> "MedGemma 1.5 4B analyses the image alongside the clinician's question.  
> It returns structured findings: no acute abnormality, heart size normal,  
> no pleural effusion — plus an impression and clarifying questions."

### 2e — Multi-Turn Refinement (1:20 – 1:40)

**[Screen: Click "Listen" to hear response → tap to add more context → refined result]**

> "The clinician can listen to the AI response via text-to-speech —  
> hands-free in the middle of an exam.  
> They can also provide follow-up context — 'Patient has a 2-week cough' —  
> and the interpretation refines itself. This is the agentic workflow loop."

---

## SCENE 3 — Final Report & Architecture (1:40 – 2:20)

### 3a — Generate Report (1:40 – 1:55)

**[Screen: Click "Final Report" → structured report appears with disclaimer]**

> "When satisfied, the clinician generates a final radiology report.  
> It includes all turns, findings, and a mandatory disclaimer:  
> AI-assisted interpretation — clinician review required."

### 3b — Architecture Overview (1:55 – 2:20)

**[Screen: Architecture diagram or code structure from README]**

> "Under the hood:  
> — The frontend is React, served as a PWA with a Service Worker for offline use.  
> — The backend is FastAPI, running MedGemma 1.5 4B for image interpretation  
>   and Whisper for speech-to-text.  
> — Sessions have a 30-minute TTL, input validation returns proper HTTP errors,  
>   and CORS is locked to configured origins.  
> — The whole stack runs in Docker with one command."

---

## SCENE 4 — Prize Alignment & Impact (2:20 – 2:50)

**[Screen: Prize alignment table from README, or custom slide]**

> "VoiceRad targets all four competition prizes:  
>  
> **Main Track** — It's a complete, deployable application solving a real clinical gap.  
> **Edge AI** — MedGemma 4B runs on a single T4 GPU; the PWA works offline on mobile.  
> **Agentic Workflow** — Voice input, image analysis, and multi-turn refinement  
>   form an autonomous diagnostic conversation loop.  
> **Novel Task** — We use MedGemma for structured radiology report generation,  
>   not just classification.  
>  
> The impact: any clinician with a phone can access AI-powered radiology  
> interpretation — in their voice, in real time, even without internet."

---

## SCENE 5 — Close (2:50 – 3:00)

**[Screen: GitHub repo page / VoiceRad title card]**

> "VoiceRad — voice-controlled radiology for the 4.5 billion.  
> Open source. Apache 2.0. Built on MedGemma.  
> Thank you."

---

## Recording Tips

1. **Resolution:** 1920×1080 minimum  
2. **Audio:** Use a quiet room; external mic preferred  
3. **Pacing:** Rehearse to land at exactly 3:00  
4. **Browser:** Use Chrome in full-screen; hide bookmarks bar  
5. **Demo data:** Have a chest X-ray image ready (e.g., NIH ChestX-ray14 sample)  
6. **Font size:** Zoom browser to 125% so text is readable  
7. **Cursor:** Use a cursor highlighter extension so viewers can follow clicks  
