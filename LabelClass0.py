import numpy
import pickle


def onehotencoding0(mode = 0):
    print("Program started"+"\n")
    if(mode == 0):
        fout_labels_class = open("E:\DataSet\data\label_class_0.dat",'w')
    elif(mode == 1):
        fout_labels_class = open("E:\DataSet\data\label_class_0_3class.dat",'w')

    set = []
    cnt1 = 0
    cnt2 = 0

    f = open('E:\DataSet\data\labels_0.dat','r')
    for val in f:
        fval = float(val)
        if fval not in set:
                set.append(fval)

    sset = numpy.array(set)
    median = numpy.median(sset)
    print(median)
    f.close()

    with open('E:\DataSet\data\labels_0.dat','r') as f:
        if(mode == 0):
            for val in f:
                if float(val) > median:
                    fout_labels_class.write(str(1) + "\n");
                    cnt1 = cnt1 +1
                else:
                    fout_labels_class.write(str(0) + "\n");
                    cnt2 = cnt2 +1
        elif(mode == 1):
            for val in f:
                if float(val) < 3:
                    fout_labels_class.write(str(0) + "\n");
                elif float(val) <6:
                    fout_labels_class.write(str(1) + "\n");
                else:
                    fout_labels_class.write(str(2) + "\n");
    set.sort()
    print(set)
    print(cnt1,cnt2)

if __name__ == '__main__':
    onehotencoding0()
    onehotencoding0(1)
