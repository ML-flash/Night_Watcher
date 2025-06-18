# Night_Watcher User Guide

This guide walks you through setting up Night_Watcher with Anaconda and using the web interface for all major tasks. Only a few command line steps are needed for installation and launching the server; everything else happens in your browser.

## 1. Install Anaconda (Python 3.8+)

1. Download the Anaconda installer for your operating system from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution).
2. Run the installer and follow the on‑screen instructions. Make sure to install a version that includes Python 3.8 or higher. Night_Watcher requires at least Python 3.8.
3. Open the Anaconda Prompt (Windows) or your terminal (macOS/Linux) and create a new environment:
   ```bash
   conda create -n nightwatcher python=3.8
   conda activate nightwatcher
   ```

The project recommends Python 3.8+ and at least 8GB RAM, as noted in the prerequisites section of the README【F:README.md†L4-L8】.

## 2. Clone the Repository

Use Git to download the project files:
```bash
git clone [your-repo-url]
cd night_watcher
```
These commands are taken from the Setup section of the README【F:README.md†L12-L16】.

## 3. Install Dependencies

With the `nightwatcher` environment activated, install the required Python packages:
```bash
pip install -r requirements.txt
```
After installing, run the provided setup check to verify your configuration:
```bash
python setup_night_watcher.py
```
These installation commands are also part of the Setup section in the README【F:README.md†L16-L22】.

## 4. Configure the Language Model

Night_Watcher can work with LM Studio or the Anthropic API. The web interface assumes LM Studio is running locally.

**Using LM Studio (Recommended)**
1. Download [LM Studio](https://lmstudio.ai/) and install it on your machine.
2. Launch LM Studio and download a compatible model (for example, Qwen2.5‑32B‑Instruct).
3. Start LM Studio's local server (default port `1234`).

These steps mirror the "Option A: Local with LM Studio" instructions in the README【F:README.md†L24-L34】. The LMStudio provider also supports streaming tokens by passing `stream=True` in API calls.

If you prefer Anthropic's hosted model, set the `ANTHROPIC_API_KEY` environment variable as shown in the README【F:README.md†L34-L36】.

## 5. Launch the Web Interface

Start the web server from the command line:
```bash
python Night_Watcher.py --web
```
After running this command, open your browser to `http://localhost:5000` to access the dashboard. These instructions come from the "Web Dashboard" section of the README【F:README.md†L52-L60】【F:README.md†L110-L119】.

## 6. Using the Dashboard

Once the dashboard is open, you can perform all major tasks without additional command line steps:

- **Collect Content**: Gather political news articles from the configured sources.
- **Analyze Content**: Run multi‑round analysis to detect manipulation techniques and extract key entities. The analysis pipeline includes seven rounds as described in the README【F:README.md†L158-L167】.
- **Review Queue**: Validate and approve analysis results.
- **Manage Sources**: Enable or disable RSS feeds and add new ones.
- **View Knowledge Graph**: Explore relationships between people, institutions, events, and patterns.
- **Vector Search**: Find articles with similar patterns.

The dashboard features are summarized in the README under "Web Dashboard"【F:README.md†L110-L119】.

## 7. Output Files

All collected and analyzed data is stored in the `data/` directory with subfolders for articles, analyses, documents, the knowledge graph, vectors, and logs. The layout is shown in the README【F:README.md†L128-L137】.

## 8. Exporting and Updating Artifacts

When you need to share or archive your results, you can create a signed bundle or integrate updates from another bundle. These operations require command line usage:
```bash
python export_artifact.py --output my_bundle.tar.gz
python update_artifact.py my_bundle.tar.gz
```
This process is documented in the "Exporting & Applying Updates" section of the README【F:README.md†L140-L156】.

## 9. Troubleshooting

If LM Studio is not responding, ensure it is running with a model loaded. You can also test the connection with `curl http://localhost:1234/v1/models`. Additional troubleshooting tips include checking log files for JSON extraction errors or adjusting your article limits if you run out of memory. These recommendations are drawn from the README's troubleshooting section【F:README.md†L172-L192】.

## 10. Security and Monitoring

Night_Watcher runs entirely on your local machine when using LM Studio. Documents are stored with cryptographic signatures, and no data is sent to external services unless you opt to use Anthropic. The system tracks patterns such as power consolidation, institutional capture, and information control as described in the README【F:README.md†L196-L208】.

## 11. Next Steps

- Keep LM Studio running whenever you analyze content.
- Periodically review the dashboard's knowledge graph and vector search results to monitor trends.
- Refer to `config.json` to adjust collection limits or source settings if needed.

By following this guide, you can manage the entire Night_Watcher workflow through the web interface, with command line use limited to initial installation and launching the server.
