# Firebase Deployment Guide for VoiceRad

This guide walks you through deploying VoiceRad to Google Firebase.

## Architecture

- **Firebase Hosting** serves the React PWA (static files from Vite build)
- **Firebase Cloud Functions (2nd gen, Python 3.11)** runs the FastAPI backend
- All `/api/*` requests are automatically routed from Hosting to Functions via rewrites

## Prerequisites

1. A Google account
2. Node.js 18+ installed locally
3. Firebase CLI installed: `npm install -g firebase-tools`

## Step 1: Create a Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com)
2. Click "Add project"
3. Name it (e.g., `voicerad-app`)
4. Follow the wizard (you can disable Google Analytics if you want)
5. Once created, note your **Project ID**

## Step 2: Update Project ID

Replace the placeholder project ID in these files:

- `.firebaserc` — change `"default": "voicerad-app"` to your project ID
- `.github/workflows/firebase-deploy.yml` — change `projectId: voicerad-app`

## Step 3: Enable Required APIs

In the [Google Cloud Console](https://console.cloud.google.com) for your project:

1. Enable **Cloud Functions API**
2. Enable **Cloud Build API**
3. Enable **Artifact Registry API**
4. Upgrade to **Blaze plan** (pay-as-you-go, required for Cloud Functions)

## Step 4: Local Deployment

```bash
# Login to Firebase
firebase login

# Build the frontend
cd frontend && npm install && npm run build && cd ..

# Copy backend into functions (needed for deploy)
cp -r backend functions/backend
touch functions/backend/__init__.py
touch functions/backend/models/__init__.py

# Deploy everything
firebase deploy
```

Your app will be live at `https://YOUR-PROJECT-ID.web.app`

## Step 5: Set Up CI/CD (GitHub Actions)

For automatic deployments on every push to main:

1. In Firebase Console, go to Project Settings > Service accounts
2. Click "Generate new private key" to download a JSON file
3. In your GitHub repo, go to Settings > Secrets > Actions
4. Add a new secret called `FIREBASE_SERVICE_ACCOUNT`
5. Paste the entire contents of the JSON file as the secret value

Now every push to `main` will automatically:
- Build the React frontend
- Copy the backend into the functions directory
- Deploy to Firebase Hosting + Functions

## Step 6: Custom Domain (Optional)

1. In Firebase Console > Hosting > "Add custom domain"
2. Follow the DNS verification steps
3. Firebase provides free SSL certificates

## Configuration Notes

### Demo Mode
The Cloud Function runs in `DEMO_MODE=1` by default since Firebase Cloud
Functions don't have GPU access. This means:
- AI model responses are simulated
- The full app UX works end-to-end
- Great for demos and evaluation

### Production with GPU Models
For production inference with MedGemma:
1. Deploy the backend to **Google Cloud Run** with a GPU-enabled container
2. Update `firebase.json` rewrites to point `/api/**` to your Cloud Run URL
3. Remove the Cloud Function (it becomes unnecessary)

### Environment Variables
Set environment variables for Cloud Functions:
```bash
firebase functions:config:set app.demo_mode="1"
```

## File Structure

```
firebase.json          # Hosting + Functions config
.firebaserc            # Project ID mapping
functions/
  main.py              # Cloud Function entry point (wraps FastAPI)
  requirements.txt     # Python dependencies
  .gitignore           # Excludes copied backend/
  backend/             # Copied from /backend at deploy time
.github/workflows/
  firebase-deploy.yml  # CI/CD workflow
```

## Troubleshooting

**"Error: No matching credentials found"**
- Run `firebase login` to authenticate

**"Billing account not configured"**  
- Upgrade to Blaze plan in Firebase Console

**Functions deploy fails with memory error**
- Increase memory in `functions/main.py` (change `MB_256` to `MB_512` or higher)

**API calls return 404**
- Verify the rewrite rules in `firebase.json` are correct
- Ensure the function name `api` matches the rewrite target
