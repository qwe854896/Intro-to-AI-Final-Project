import tkinter

mouseStatus = "up"
xold, yold = None, None
lineWidth = 1
lineColor = "black"

def updateWidth(value):
    global lineWidth
    lineWidth = value

def main():
    app = tkinter.Tk()
    app.title("Intro to AI Final Project")
    app.geometry("800x800")
    app.configure(background='gray')

    drawingArea = tkinter.Canvas(app,width=600,height=600)
    drawingArea.place(relx = 0.5, rely = 0.5, anchor = tkinter.CENTER)
    drawingArea.bind("<Motion>", motion)
    drawingArea.bind("<ButtonPress-1>", mouseDown)
    drawingArea.bind("<ButtonRelease-1>", mouseUp)

    BOTTOM = tkinter.Frame(app)
    BOTTOM.configure(background = 'gray')
    BOTTOM.pack(side = tkinter.BOTTOM)

    runButton = tkinter.Button(BOTTOM, fg="green", text="Run", command = lambda:getter(drawingArea, isAnime, isConcise, isClassical), activebackground = 'red')
    runButton.pack(side = tkinter.LEFT)

    clearButton = tkinter.Button(BOTTOM, fg="green", text="Clear", command = lambda:clear(drawingArea))
    clearButton.pack(side = tkinter.LEFT, padx = 3)

    changeColorButton = tkinter.Button(BOTTOM, fg = "green", text = "Change color", command = lambda:changeColor())
    changeColorButton.pack(side = tkinter.LEFT, padx = 3)

    widthScale = tkinter.Scale(BOTTOM, label = 'width', from_ = 1, to = 10, resolution = 0.5, orient = tkinter.HORIZONTAL, showvalue = 0, command = updateWidth)
    widthScale.pack(side = tkinter.LEFT, padx = 3)

    LEFT = tkinter.Frame(app)
    LEFT.configure(background = 'gray')
    LEFT.pack(side = tkinter.LEFT)

    isAnime = tkinter.IntVar()
    tkinter.Checkbutton(LEFT, text = "Anime", variable = isAnime, width = 9).pack()
    isConcise = tkinter.IntVar()
    tkinter.Checkbutton(LEFT, text = "Concise", variable = isConcise, width = 9).pack()
    isClassical = tkinter.IntVar()
    tkinter.Checkbutton(LEFT, text = "Classical", variable = isClassical, width = 9).pack()


    def changeColor():
        from tkinter.colorchooser import askcolor
        global lineColor
        lineColor = askcolor(title='Choose your color')[1]

    def clear(widget):
        widget.delete("all")

    def getter(widget, isAnime, isConcise, isClassical):
        from PIL import ImageGrab
        import datetime
        x = app.winfo_rootx() + widget.winfo_x()
        y = app.winfo_rooty() + widget.winfo_y()
        x1 = x * 2 + widget.winfo_width() * 2
        y1 = y * 2 + widget.winfo_height() * 2
        fileName = str(datetime.datetime.now()).replace(":", "-")
        ImageGrab.grab().crop((x * 2, y * 2, x1, y1)).save(fileName + ".png")
        print(isAnime.get(), isConcise.get(), isClassical.get())

    app.mainloop()

def mouseDown(event):
    global mouseStatus
    mouseStatus = "down"
def mouseUp(event):
    global mouseStatus, xold, yold
    mouseStatus = "up"
    xold = None
    yold = None

def motion(event):
    if mouseStatus == "down":
        global xold, yold
        if xold is not None and yold is not None:
            event.widget.create_line(xold, yold, event.x, event.y, smooth = tkinter.TRUE, fill = lineColor, width = lineWidth)
        xold = event.x
        yold = event.y
if __name__ == "__main__":
    main()
