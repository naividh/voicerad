const CACHE = "voicerad-v1";
const PRECACHE = ["/", "/index.html"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE)));
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
  );
  self.clients.claim();
});

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(
    caches.match(e.request).then(
      (cached) =>
        cached ||
        fetch(e.request)
          .then((res) => {
            if (res && res.status === 200) {
              const clone = res.clone();
              caches.open(CACHE).then((c) => c.put(e.request, clone));
            }
            return res;
          })
          .catch(() => caches.match(e.request))
    )
  );
});

// Background sync for offline interpretations
self.addEventListener("sync", (e) => {
  if (e.tag === "sync-interpretations") {
    e.waitUntil(syncPending());
  }
});

async function syncPending() {
  const db = await new Promise((resolve, reject) => {
    const req = indexedDB.open("VoiceRadDB", 1);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onupgradeneeded = (ev) => {
      ev.target.result.createObjectStore("pending", { keyPath: "id", autoIncrement: true });
    };
  });

  const tx = db.transaction("pending", "readwrite");
  const store = tx.objectStore("pending");
  const items = await new Promise((r) => {
    const req = store.getAll();
    req.onsuccess = () => r(req.result);
  });

  for (const item of items) {
    try {
      const res = await fetch("/api/interpret/start-session", { method: "POST", body: item.formData });
      if (res.ok) store.delete(item.id);
    } catch {
      // will retry on next sync
    }
  }
}
