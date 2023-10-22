# Talk with Document using LLM's

## Author :man:
:scroll: Original article is [here](https://pub.towardsai.net/talk-to-your-documents-as-pdfs-txts-and-even-web-pages-717b5af8f48c)

:factory: Original github repo is [here](https://github.com/damiangilgonzalez1995/TalkDocument)

## How to run
To run streamlit application
```bash
streamlit run src/Front.py
```

To test backend logic
```bash
python src/main.py
```

## Repository
```
.
├── data                    # Data files for testing
│   └── ...
├── images                  # Images for Streamlit application
│   └── ...
├── src                     # App sources
│   ├── helper              
│   ├── pages               # Folder for Streamlit Multipage app
│   │   ├── 1_step_1...py
│   │   ├── 2_step_2...py
│   │   └── ...
│   ├── Front.py            # Streamlit "entrypoint" file
│   ├── main.py             # Entrypoint for backend logic testing
│   ├── TalkDocument.py     # Backend Logic
│   └── _init.py_             
└── ...
```
