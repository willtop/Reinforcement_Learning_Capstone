import pywinauto as pywa

handle = pywa.findwindows.find_windows(title='Android Emulator - WVGA:5554')[0]
print(handle)

app = pywa.application.Application().connect(handle=handle)
print(app)
print(app.windows_())
i = 0
for dialog in app.windows_():
    im = dialog.CaptureAsImage()
    if im is not None:
        im.save(str(i) + '.png')
    i += 1