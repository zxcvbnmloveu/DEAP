def onehotencoding1(mode = 0):
    print("Program started"+"\n")
    if(mode == 0):
        fout_labels_class = open("E:\DataSet\data\label_class_1.dat",'w')
    elif(mode == 1):
        fout_labels_class = open("E:\DataSet\data\label_class_1_3class.dat",'w')
    with open('E:\DataSet\data\labels_1.dat','r') as f:
        if(mode == 0):
            for val in f:
                if float(val) > 4.5:
                    fout_labels_class.write(str(1) + "\n");
                else:
                    fout_labels_class.write(str(0) + "\n");
        elif(mode == 1):
            for val in f:
                if float(val) < 3:
                    fout_labels_class.write(str(0) + "\n");
                elif float(val) <6:
                    fout_labels_class.write(str(1) + "\n");
                else:
                    fout_labels_class.write(str(2) + "\n");

if __name__ == '__main__':
    onehotencoding1(1)
