
with open(r'f:\YOLOv10_Bike_Detection\ui_base.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('&gt;', '>')
content = content.replace('&lt;', '<')
content = content.replace('&amp;', '&')
content = content.replace('&quot;', '"')
content = content.replace('&#39;', "'")

with open(r'f:\YOLOv10_Bike_Detection\ui_base.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('HTML转义字符修复完成！')
