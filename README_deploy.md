# Deploying these 3 new pages to bhuvi-ai-lab.streamlit.app

These files match the multipage pattern your existing app already uses
(PRD_Generator, Jira_Story_Generator, RAG_Knowledge_Assistant).

## Steps

1. Copy the 3 files below into the `pages/` folder of your existing
   `bhuvi-ai-lab` Streamlit repo:
   - `Mobile_App_Review_Analyzer.py`
   - `AI_Product_Copilot.py`
   - `AI_Predictive_Maintenance_Dashboard.py`

2. Add these packages to your repo's `requirements.txt` (merge with what's
   already there, don't duplicate):

   ```
   streamlit
   pandas
   numpy
   scikit-learn
   plotly
   openai
   ```

   (`openai` is only needed for the optional "polish with AI" button in the
   Product Copilot — the apps work without it.)

3. Commit and push. Streamlit Community Cloud will redeploy automatically,
   and the new pages will be live at:
   - `https://bhuvi-ai-lab.streamlit.app/Mobile_App_Review_Analyzer`
   - `https://bhuvi-ai-lab.streamlit.app/AI_Product_Copilot`
   - `https://bhuvi-ai-lab.streamlit.app/AI_Predictive_Maintenance_Dashboard`

4. Those exact URLs are already wired into the updated `index.html` — no
   further edits needed once deployed. If your page filenames or app name
   differ, update the `href` values in the "AI Lab" and "Case Studies"
   sections of `index.html` to match.

## Local test before deploying

```
pip install streamlit pandas numpy scikit-learn plotly
streamlit run Mobile_App_Review_Analyzer.py
```
