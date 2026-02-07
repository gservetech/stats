import streamlit as st


def render_tab_friday_playbook():
    st.markdown("""
## 📜 Friday Gamma Rulebook (READ BEFORE TRADING)

*Fridays are GAMMA days — not conviction days.*

### 🔑 Core Truths
1. *Walls control price until they break*
2. *Moving walls are targets, not resistance*
3. *Spreads near walls lose on Fridays*
4. *Long gamma beats being right*
5. *No break = no trade*

---

### 🧭 Decision Checklist
Ask these in order — do NOT skip:

*1️⃣ Is price between Call & Put walls?*
- YES → Expect chop / pin → Do nothing
- NO  → Momentum possible → go to step 2

*2️⃣ Is price near the Magnet (max OI)?*
- YES → Pin risk is high → Wait
- NO  → go to step 3

*3️⃣ Is flow proxy (Volume/OI) building near spot?*
- YES → Dealers may flip → Prepare for break
- NO  → Structure still in control

*4️⃣ Did a wall BREAK with follow-through?*
- YES → Buy *single option, next-week expiry*
- NO  → Stand down

---

### ✅ What to Trade on Fridays
✔ Single calls/puts  
✔ Next-week expiry (keep gamma, avoid 0DTE traps)  
✔ Enter AFTER confirmation  
✔ Exit when momentum slows  

---

### 🚫 What NOT to Trade on Fridays
✘ Tight debit spreads  
✘ Credit spreads near walls  
✘ “It’s overextended” fades  
✘ Holding to expiration  
✘ Fighting wall migration  

---

### 🧠 If confused → DO NOTHING
Not trading is a position.
""")
