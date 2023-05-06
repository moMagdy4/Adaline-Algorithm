
from tkinter import *
from tkinter.ttk import Separator
from tkinter import ttk
import pandas as pd

import matplotlib.pyplot as plt
import Helper_functions as model


def Data_from_gui_to_Model():
    data = pd.read_csv('penguins.csv')
    Feature1 = feature1_Value.get()
    Feature2 = feature2_Value.get()

    Class1 = class1_Value.get()
    Class2 = class2_Value.get()

    le_str = learningRate_TextField.get()
    le_rate = learningRate_TextField.getdouble(le_str)

    epochs_str = number_Of_Epochs_TextField.get()
    epochs = number_Of_Epochs_TextField.getint(epochs_str)

    thresh_str = number_Of_Epochs_TextField.get()
    thresh_hold = number_Of_Epochs_TextField.getint(thresh_str)

    bias = biasCheckBox.get()
    w, b, acc, ms = model.Model(data, Class1, Class2, Feature1,
                                Feature2, le_rate, epochs, bias, thresh_hold)
    open_popup(w, b, acc)

# plot 3 classes


def plotGraph(X_AXIS, Y_AXIS):
    data_to_plot_classes = pd.read_csv('penguins.csv')
    data_to_plot = model.Pre_processing(data_to_plot_classes)
    data_class1 = data_to_plot.loc[data_to_plot['species'].isin(['Adelie'])]
    data_class2 = data_to_plot.loc[data_to_plot['species'].isin([
        'Gentoo'])]
    class3DataFrame = data_to_plot.loc[data_to_plot['species'].isin([
                                                                    'Chinstrap'])]

    plt.figure('Graph')
    plt.cla()
    plt.scatter(data_class1[X_AXIS], data_class1[Y_AXIS], color='red')
    plt.scatter(data_class2[X_AXIS], data_class2[Y_AXIS], color='blue')
    plt.scatter(class3DataFrame[X_AXIS],
                class3DataFrame[Y_AXIS], color='green')
    plt.xlabel(X_AXIS)
    plt.ylabel(Y_AXIS)
    plt.show()


# GUI
features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "gender",
    "body_mass_g",
]

species = [
    "Adelie",
    "Gentoo",
    "Chinstrap",
]


def open_popup(w, b, acc):
    top = Toplevel(MAIN)
    top.geometry("350x500")
    top.title("Values")
    # weight&bias
    weight_text_label = Label(top, text="WEIGHT IS:")
    weight_text_label.pack()
    weight_value = str(w)
    weight_value_label = Label(top, text=weight_value)
    weight_value_label.pack()
    bias_text_label = Label(top, text="BIAS IS:")
    bias_text_label.pack()
    bias_value = str(b)
    bias_value_label = Label(top, text=bias_value)
    bias_value_label.pack()
    # accurcy
    accurcy_text_label = Label(top, text="ACCURCY IS:")
    accurcy_text_label.pack()
    accurcy_value = str(acc)
    accurcy_value_label = Label(top, text=accurcy_value)
    accurcy_value_label.pack()


if __name__ == '__main__':

    # Main window
    MAIN = Tk()
    MAIN.title('Task NN 1')
    MAIN.geometry("350x500")

    ########################
    # Select Features
    Header_features = Label(MAIN, text="Select 2 Features")
    Header_features.pack()
    # 1
    feature1_Value = StringVar()
    feature1_Value.set("Feature 1")
    feature1_DropMenu = OptionMenu(MAIN, feature1_Value, *features)
    feature1_DropMenu.pack()
    # 2
    feature2_Value = StringVar()
    feature2_Value.set("Feature 2")
    feature2_DropMenu = OptionMenu(MAIN, feature2_Value, *features)
    feature2_DropMenu.pack()

    ##########################
    # Select Classes
    class_Header = Label(MAIN, text='Select 2 Classes')
    class_Header.pack()
    # 1
    class1_Value = StringVar()
    class1_Value.set("Class1")
    class1_DropMenu = OptionMenu(MAIN, class1_Value, *species)
    class1_DropMenu.pack()
    # 2
    class2_Value = StringVar()
    class2_Value.set("Class2")
    class2_DropMenu = OptionMenu(MAIN, class2_Value, *species)
    class2_DropMenu.pack()

    ###########################
    # Add Learning Rate
    learningRate_Header = Label(MAIN, text="Add Learning Rate")
    learningRate_Header.pack()

    learningRate_TextField = ttk. Entry(MAIN, width=20)
    learningRate_TextField.pack()

    ###############################

    # Add Epochs
    number_Of_Epochs_Header = Label(
        MAIN, text="Add Number Of Epochs")
    number_Of_Epochs_Header.pack()
    number_Of_Epochs_TextField = Entry(MAIN, width=20)
    number_Of_Epochs_TextField.pack()
    # Add threshhold
    number_Of_threshhold_Header = Label(
        MAIN, text="Add Number Of thresh hold")
    number_Of_threshhold_Header.pack()
    number_Of_threshhold_TextField = Entry(MAIN, width=20)
    number_Of_threshhold_TextField.pack()

    # Select Bias
    biasCheckBox = IntVar()
    checkbox = Checkbutton(MAIN, text='Bias',
                           variable=biasCheckBox)
    checkbox.pack()

    # Start Classification

    button = Button(MAIN, text="Start Modeling",
                    command=Data_from_gui_to_Model)
    button.pack()

    sep = Separator(MAIN, orient='horizontal')
    sep.pack(fill='x')

    # Select feature to plot graph
    plotGraphBtn2 = Button(MAIN, text='Plot Graph Of 3 classes',
                           command=lambda: plotGraph(feature1_Value.get(), feature2_Value.get())).pack()

    MAIN.mainloop()
