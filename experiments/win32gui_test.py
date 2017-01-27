import win32gui, win32ui, win32con
from  PIL import Image
from  PIL import ImageGrab

from ctypes import windll


def getWindowHandle(name):
    toplist, winlist = [], []

    def enum_cb(hwnd, results):
        winlist.append((hwnd, win32gui.GetWindowText(hwnd)))

    win32gui.EnumWindows(enum_cb, toplist)

    name = name.lower()
    hwnd_title_tuple = [(hwnd, title) for hwnd, title in winlist if name in title.lower()]
    # just grab the hwnd for first window matching hwnd_title_tuple
    hwnd_title_tuple = hwnd_title_tuple[0]
    hwnd = hwnd_title_tuple[0]

    return hwnd


class Window():
    def __init__(self, hwnd = None):
        if not hwnd: return

        l, t, r, b   = win32gui.GetClientRect(hwnd)
        sl, st, _, _ = win32gui.GetWindowRect(hwnd)
        cl, ct       = win32gui.ClientToScreen(hwnd, (l, t))

        self.size     = (r - l, b - t)
        self.position = (cl - sl, ct - st)

        hDC   = win32gui.GetWindowDC(hwnd)
        self.windowDC  = win32ui.CreateDCFromHandle(hDC)
        self.newDC = self.windowDC.CreateCompatibleDC()

        #win32gui.ReleaseDC(hwnd, hDC)

        self.bitmap = win32ui.CreateBitmap()
        self.bitmap.CreateCompatibleBitmap(self.windowDC, self.size[0], self.size[1])
        self.newDC.SelectObject(self.bitmap)

    def __del__(self):
        self.newDC.DeleteDC()
        self.windowDC.DeleteDC()
        del self.bitmap

    def screenshot(self):
        self.newDC.BitBlt((0, 0), self.size, self.windowDC, self.position, win32con.SRCCOPY)
        self.bitmap.Paint(self.newDC)

        bmpinfo = self.bitmap.GetInfo()
        bmpstr  = self.bitmap.GetBitmapBits(True)
        im = Image.frombuffer('RGB', self.size, bmpstr, 'raw', 'BGRX', 0, 1)
        try:
            print(im)
            im.save('z.png', 'PNG')
        except IOError:
            return


def main():
    handle = getWindowHandle("Android Emulator - WVGA:5554")
    print(handle)
    if not handle: return

    window = Window(handle)
    window.screenshot()


def main2():
    hwnd = win32gui.FindWindow(None, 'Android Emulator - WVGA:5554')

    # Change the line below depending on whether you want the whole window
    # or just the client area.
    # left, top, right, bot = win32gui.GetClientRect(hwnd)
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bot - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)

    # Change the line below depending on whether you want the whole window
    # or just the client area.
    # result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)
    print(result)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    im = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result == 1:
        # PrintWindow Succeeded
        im.save("z.png")

if __name__ == "__main__":
    # main()
    # main2()

    # hwnd = win32gui.FindWindow(None, 'Calculator')
    #
    # hdc = win32gui.GetDC(hwnd)
    # hdcMem = win32gui.CreateCompatibleDC(hdc)
    #
    # hbitmap = win32ui.CreateBitmap()
    # hbitmap = win32gui.CreateCompatibleBitmap(hdcMem, 500, 500)
    #
    # win32gui.SelectObject(hdcMem, hbitmap)
    #
    # a = windll.user32.PrintWindow(hwnd, hdcMem, 0)
    # print(a)

    hwnd = win32gui.FindWindowEx(None, 0, None, 'Android Emulator - WVGA:5554')

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveDC.SetWindowOrg((13, 151))
    saveBitMap = win32ui.CreateBitmap()

    saveBitMap.CreateCompatibleBitmap(mfcDC, 1920, 1080)
    saveDC.SelectObject(saveBitMap)
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    im = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    if result != 0:
        im.save("z.bmp")
    else:
        print('WTFFFFFFFFFFFFF')


    # toplist, winlist = [], []
    #
    # def enum_cb(hwnd, results):
    #     winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
    #
    # win32gui.EnumWindows(enum_cb, toplist)
    #
    # print(winlist)
    # firefox = [(hwnd, title) for hwnd, title in winlist if 'Android Emulator - WVGA:5554'.lower() in title.lower()]
    # # just grab the hwnd for first window matching firefox
    # firefox = firefox[0]
    # hwnd = firefox[0]
    #
    # win32gui.SetForegroundWindow(hwnd)
    # bbox = win32gui.GetWindowRect(hwnd)
    #
    # print(bbox)
    # img = ImageGrab.grab(bbox)
    # print(img)
    # img.show()
