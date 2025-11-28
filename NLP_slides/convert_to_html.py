import markdown

with open('DIDACTIC_PRESENTATION_FRAMEWORK.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

html_template = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Didactic Presentation Framework</title>
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px;
    line-height: 1.6;
    color: #333;
    background-color: #fff;
}
h1, h2, h3 { color: #2c3e50; margin-top: 30px; }
h1 { font-size: 2.5em; border-bottom: 3px solid #3498db; padding-bottom: 15px; }
h2 { font-size: 2em; margin-top: 50px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
h3 { font-size: 1.5em; border-bottom: 1px solid #bdc3c7; padding-bottom: 8px; }
code {
    background-color: #f4f4f4;
    padding: 3px 6px;
    border-radius: 3px;
    font-family: "Courier New", Consolas, monospace;
    font-size: 0.9em;
}
pre {
    background-color: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    overflow-x: auto;
    margin: 20px 0;
}
pre code { background-color: transparent; padding: 0; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 25px 0;
    box-shadow: 0 2px 3px rgba(0,0,0,0.1);
}
th, td { border: 1px solid #ddd; padding: 12px 15px; text-align: left; }
th { background-color: #3498db; color: white; font-weight: bold; }
tr:nth-child(even) { background-color: #f9f9f9; }
tr:hover { background-color: #ecf0f1; }
ul, ol { margin: 15px 0; padding-left: 35px; }
li { margin: 10px 0; }
blockquote {
    border-left: 4px solid #3498db;
    margin: 25px 0;
    padding-left: 20px;
    color: #555;
    font-style: italic;
    background-color: #f9f9f9;
    padding: 15px 15px 15px 20px;
}
hr { border: none; border-top: 2px solid #ecf0f1; margin: 50px 0; }
strong { color: #2c3e50; }
.checkmark { color: #27ae60; font-weight: bold; }
</style>
</head>
<body>
""" + html_content + """
</body>
</html>"""

with open('DIDACTIC_PRESENTATION_FRAMEWORK.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

print("HTML file created successfully: DIDACTIC_PRESENTATION_FRAMEWORK.html")