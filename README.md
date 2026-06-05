# 🎓 GGU CSITmitra Chatbot

A smart, context-aware Helpdesk for **CSIT programs at Guru Ghasidas University (GGU)**.
Built using **Python**, **NLTK**, and **fuzzy matching**, this chatbot provides detailed academic, faculty, and admission-related information.
It also saves the chat data in a separate folder by asking the user whether they want to save the data or not.
It uses Pure python code and currently it has limited data only about the department.

## 🚀 - Ideas to implment later:

  1). Creating webpage for this page.
  2). Adding to whatsapp
  3). Adding an free opem source AI api for enabling knowledegble ai chatbot.
  4). Adding the data to University level.
  5). Recreating the whole code in a more modular way with OOPs concepts instead of boolean logic.
  6). Adding a feedback system for users to rate the helpfulness of responses, and using that data to improve the bot over time.
  7). Recreating the whole .json data file by using binary trees for better data storage and retrieval.


---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install nltk colorama
```

### 2. Run the Chatbot

```bash
python ggu_csit_chatbot.py
```

---

## ✨ Features

| Feature                    | Description                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------- |
| 📁 **JSON Data File**      | All chatbot data is stored in `ggu_data.json`, making it easy to update without modifying code    |
| 📚 **Detailed Courses**    | Semester-wise syllabus for MCA, M.Sc. CS, M.Sc. IT, Integrated programs, and Ph.D.                |
| 👩‍🏫 **Faculty Profiles** | Includes qualifications, subjects taught, ORCID, Google Scholar links, patents, and joining dates |
| 💰 **Fee Breakdown**       | Complete fee structure with per-semester, total cost, and hostel fees (boys & girls)              |
| 🔍 **Fuzzy Matching**      | Handles typos like *“facuty”*, *“cours”*, *“addmission”*                                          |
| 🧠 **Context Memory**      | Remembers previous queries (e.g., “MCA” → “what about fees?”)                                     |
| 🎯 **Specific Lookups**    | Instantly fetch faculty details (e.g., “Tell me about Dr. Shrivas”)                               |

---

## 🧠 How It Works

The chatbot uses:

* **Python** for core development
* **NLTK** for natural language processing
* Rule-based + fuzzy matching logic for understanding user queries
* A structured JSON database (`ggu_data.json`) for fast and flexible responses

---

## 💬 Example Queries

Try asking:

```
Tell me about MCA
MCA syllabus
What are the fees
Who is Dr. Babita Majhi
Tell me about Shrivas
What is the exam pattern
PhD admission
```

---

## 📂 Project Structure

```
.
├── ggu_csit_chatbot.py   # Main chatbot script
├── ggu_data.json         # Data source (courses, faculty, fees, etc.)
├── README.md             # Project documentation
```

---

## 🔧 Customization

* Edit `ggu_data.json` to:

  * Add new courses
  * Update faculty profiles
  * Modify fee structures
* No need to change the Python code for most updates

---

## 📌 Requirements

* **Python** 3.x
* Libraries:

  * **NLTK**
  * **Colorama**

---

## 🌟 Future Improvements

* Web-based interface (Flask/Django)
* Voice input support
* Integration with university APIs
* Multi-language support

---

## 📜 License

This project is for educational purposes. You can modify and extend it as needed.
