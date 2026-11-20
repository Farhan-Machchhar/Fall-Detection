# 🎬 3.5-Minute Demonstration Video Script: CareGuard Fall Detection System

> **Project:** AI-03 — Fall Detection from Phone and Wearable Sensors  
> **Organization:** TheCarePal  
> **Speaker/Presenter:** Farhan (AI/ML Engineering Intern)  
> **Target Duration:** 3 minutes 30 seconds  
> **Format:** Screen Recording with Voiceover

---

## Video Outline & Timing Summary

| Segment | Time | Focus | Visual Action on Screen |
|---|---|---|---|
| **1. Introduction & The Problem** | 0:00 – 0:35 | Problem context & project objective | Title slide / Overview tab in Streamlit |
| **2. System Architecture & Dataset** | 0:35 – 1:05 | UMAFall dataset & ML approach | Architecture diagram in About tab & code snippet |
| **3. Dual Detection & Sensor Fusion** | 1:05 – 1:45 | Both devices active & Sensor Monitor | Live accelerometer/gyroscope charts & Detection tab |
| **4. Device Fallback Scenarios** | 1:45 – 2:25 | Disconnection & Not-Worn fallback | Toggling device switches in sidebar & live updates |
| **5. Emergency Alert Countdown** | 2:25 – 3:00 | 30s timer, cancellation & notification | Alert System tab with countdown & caregiver message |
| **6. Evaluation & Conclusion** | 3:00 – 3:30 | Model metrics & closing summary | Model Performance tab with confusion matrices |

---

## Detailed Scene-by-Scene Script

### Segment 1: Introduction & The Problem (0:00 – 0:35)
**Visual on Screen:**
- Streamlit application open on the **Overview** tab.
- Browser URL: `http://localhost:8501`.
- Clean dark-blue interface with CareGuard header and system status badge.

**Audio / Voiceover:**
> *"Hello everyone! I'm Farhan, and today I'm demonstrating **CareGuard**, an AI-powered fall-detection system developed for TheCarePal.
> 
> Falls among elderly individuals are a leading cause of severe injury and loss of independence worldwide. Traditional alert pendants require the user to manually press a button, which rarely happens during a traumatic fall.
> 
> On the other hand, automatic systems relying on a single device fail when a smartwatch is charging or when a smartphone is left on a desk. CareGuard solves this by utilizing **dual motion sensing from both smartphone and wearable devices**, coupled with **autonomous fallback and sensor fusion**."*

---

### Segment 2: Machine Learning Approach & UMAFall Dataset (0:35 – 1:05)
**Visual on Screen:**
- Navigate to the **About** tab.
- Scroll down to highlight the **System Architecture** diagram and **Dataset Attribution**.
- Briefly show VS Code with `models/phone_model.onnx` and `models/wearable_model.onnx`.

**Audio / Voiceover:**
> *"To build CareGuard, we trained our models on the public **UMAFall dataset**, comprising over 740 motion recordings from 17 subjects across both falls and activities of daily living.
> 
> We implement a 2-second sliding-window pipeline extracting 66 time-domain statistical and kinematic features across accelerometer and gyroscope channels.
> 
> We trained independent **Random Forest classifiers** and comparison **Multilayer Perceptrons** with strict participant-level splitting to prevent data leakage. The models were exported to **ONNX format**, delivering sub-millisecond edge inference on CPU."*

---

### Segment 3: Dual-Device Detection & Sensor Monitoring (1:05 – 1:45)
**Visual on Screen:**
- Navigate to the **Sensor Monitor** tab.
- From the scenario dropdown, select **'Forward Fall (Subj 02, Trial 5)'**.
- Click **'Advance Window'** or **'Evaluate Next Window'**.
- Point out the 3-axis accelerometer and gyroscope graphs showing acceleration peaks.
- Switch to the **Detection** tab showing **Fall Probability: 95.4%** and Source: **'Sensor Fusion (weighted_fusion)'**.

**Audio / Voiceover:**
> *"Now let's see CareGuard in action. Here in the **Sensor Monitor**, we can observe real-time 3-axis accelerometer and gyroscope motion streams. 
> 
> When both the smartphone in the right pocket and the wrist wearable are active, CareGuard feeds both streams into their respective ONNX models and combines their probabilities using weighted sensor fusion.
> 
> As you can see, when a forward fall occurs, both sensors detect the sharp impact spike, producing a combined fall probability of over 95%."*

---

### Segment 4: Autonomous Device Fallback Logic (1:45 – 2:25)
**Visual on Screen:**
- In the sidebar, toggle **'Wearable Worn'** to **OFF** (simulating taking off the watch).
- Click **'Evaluate Next Window'**. Point out the status change: **'Phone Only (Wearable Not Worn Fallback)'**.
- Next, toggle **'Wearable Connected'** to **OFF**. Status changes to **'Phone Only (Wearable Disconnected Fallback)'**.
- Toggle **'Phone Sensor Active'** to **OFF** while turning the wearable back **ON**. Status updates to **'Wearable Only (Phone Unavailable Fallback)'**.
- Turn both **OFF**: Status displays **'Sensor Unavailable'**.

**Audio / Voiceover:**
> *"A key strength of CareGuard is its **autonomous 5-row fallback matrix**.
> 
> What happens if the user removes their smartwatch to take a shower? In the sidebar, I'll switch 'Wearable Worn' to off. Instantly, the detection mode switches to **Phone Only Fallback**. The user remains protected without interruption.
> 
> If the user walks out of Bluetooth range, it falls back to the phone. If the phone is left on a bedside table while they walk around with the watch, the system falls back to the **Wearable Only model**.
> 
> And if both devices are disconnected, the dashboard clearly warns that sensors are unavailable."*

---

### Segment 5: Non-Blocking Alert Countdown & Caregiver Simulation (2:25 – 3:00)
**Visual on Screen:**
- Navigate to the **Alert System** tab.
- Click **'Simulate Immediate Fall Alert'** (or trigger a real fall from the scenario).
- The red alert banner appears: **'🚨 FALL DETECTED — EMERGENCY COUNTDOWN ACTIVE'**.
- Show the 30-second countdown ticking down smoothly: `⏱ 28s`, `⏱ 27s`...
- Click **'🛑 I AM OK — CANCEL ALERT'**. Show the green **'Alert Cancelled by User'** confirmation.
- Trigger another alert and let it run to `⏱ 00s`. Show the blue banner: **'📱 CAREGIVER NOTIFICATION DISPATCHED (SIMULATED)'**.

**Audio / Voiceover:**
> *"When a fall is detected, CareGuard activates the emergency alert system.
> 
> A prominent 30-second countdown begins immediately. Crucially, this countdown is non-blocking. If the user tripped but is uninjured or if it was an accidental false trigger, they can simply tap **'I AM OK'**, instantly cancelling the alert.
> 
> However, if the user is unconscious or incapacitated and cannot respond, the timer reaches zero. At that moment, CareGuard automatically triggers an emergency notification, dispatching a simulated SMS and push alert to registered family members and caregivers."*

---

### Segment 6: Evaluation Results & Conclusion (3:00 – 3:30)
**Visual on Screen:**
- Navigate to the **Model Performance** tab.
- Scroll through the metrics table (Sensitivity, Specificity, F1, False Alarms/hr).
- Highlight the **Phone Confusion Matrix** and **Wearable Confusion Matrix**.
- Point out the edge latency table (<0.4 ms/window).
- Return to the Overview tab for closing remarks.

**Audio / Voiceover:**
> *"On held-out participants, our Phone Random Forest model achieves **93.3% accuracy, 92.8% sensitivity**, and a false alarm rate of just 2.3 events per hour of continuous daily activity. Our Wearable model achieves **88.9% accuracy and 87.3% sensitivity**.
> 
> Tested through 98 automated unit and end-to-end requirement tests, CareGuard demonstrates that intelligent multi-device sensor fusion and autonomous fallback can provide dependable, responsive fall protection for elderly care.
> 
> Thank you for your time, and I look forward to your questions!"*

---

## Presenter Preparation Checklist
- [x] Streamlit app running locally (`streamlit run app.py`).
- [x] Web browser zoomed to 100% or 110% for crisp UI readability.
- [x] Audio input checked for clear vocal capture without background noise.
- [x] Dataset files available at `d:\Farhan\Assignment\Dataset`.
- [x] Pytest suite verified green (`pytest tests/ -v`).
