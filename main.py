import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("WXAgg")
import matplotlib.pyplot as plot

titanic_dataset = pd.read_csv("titanic.csv")

titanic_dataset = titanic_dataset.fillna(0)

titanic_dataset = titanic_dataset.astype(
    {"pclass": 'int64', "survived": 'int64', "age": 'int64', "sibsp": 'int64', "parch": 'int64', "fare": 'int64'})


def stacked_bar_plot_gender_vs_survivor():
    female_survived = len(titanic_dataset[(titanic_dataset["sex"] == 'female') & (titanic_dataset['survived'] == 1)])
    female_unsurvived = len(titanic_dataset[(titanic_dataset["sex"] == 'female') & (titanic_dataset['survived'] == 0)])
    male_survived = len(titanic_dataset[(titanic_dataset["sex"] == 'male') & (titanic_dataset['survived'] == 1)])
    male_unsurvived = len(titanic_dataset[(titanic_dataset["sex"] == 'male') & (titanic_dataset['survived'] == 0)])
    label = ['Male', 'Female']

    male = [male_survived, male_unsurvived]
    female = [female_survived, female_unsurvived]

    fig, axis = plot.subplots(figsize=(4, 4))
    plot.subplots_adjust(left=0.15)

    axis.bar(label, male, width=0.35, label='Survived')
    axis.bar(label, female, width=0.35, bottom=male, label='Unsurvived')

    axis.set_ylabel('Number of Persons')
    axis.set_xlabel('Gender')
    axis.set_title('Survivors and Unsurvivors Group by Gender')
    axis.legend()

    plot.savefig('stacked_bar_plot_gender_vs_survivor.svg')
    plot.show()


def stacked_bar_plot_survivor_vs_class():
    survived_of_class = []
    unsurvived_of_class = []

    survived_of_class.append(len(titanic_dataset[(titanic_dataset['survived'] == 1) & (
            titanic_dataset['pclass'] == 1)]))
    survived_of_class.append(len(titanic_dataset[(titanic_dataset['survived'] == 1) & (
            titanic_dataset['pclass'] == 2)]))
    survived_of_class.append(len(titanic_dataset[(titanic_dataset['survived'] == 1) & (
            titanic_dataset['pclass'] == 3)]))

    unsurvived_of_class.append(len(titanic_dataset[(titanic_dataset['survived'] == 0) & (
            titanic_dataset['pclass'] == 1)]))
    unsurvived_of_class.append(len(titanic_dataset[(titanic_dataset['survived'] == 0) & (
            titanic_dataset['pclass'] == 2)]))
    unsurvived_of_class.append(len(titanic_dataset[(titanic_dataset['survived'] == 0) & (
            titanic_dataset['pclass'] == 3)]))
    label = ['1st', '2nd', '3rd']

    fig, axis = plot.subplots(figsize=(4, 4))
    plot.subplots_adjust(left=0.15)

    axis.bar(label, survived_of_class, width=0.35, label='Survived')
    axis.bar(label, unsurvived_of_class, width=0.35, bottom=survived_of_class, label='Unsurvived')

    axis.set_ylabel('Number of Persons')
    axis.set_xlabel('Class')
    axis.set_title('Survivors and Unsurvivors Group by Class')
    axis.legend()

    plot.savefig('stacked_bar_plot_survivor_vs_class.svg')
    plot.show()


def grouped_bar_chart_gender_vs_survivor_of_Pclass():
    male_survived = []
    female_survived = []

    male_survived.append(len(titanic_dataset[(titanic_dataset["sex"] == 'male') & (titanic_dataset['survived'] == 1) & (
                titanic_dataset['pclass'] == 1)]))
    male_survived.append(len(titanic_dataset[(titanic_dataset["sex"] == 'male') & (titanic_dataset['survived'] == 1) & (
                titanic_dataset['pclass'] == 2)]))
    male_survived.append(len(titanic_dataset[(titanic_dataset["sex"] == 'male') & (titanic_dataset['survived'] == 1) & (
                titanic_dataset['pclass'] == 3)]))

    female_survived.append(len(titanic_dataset[
                                   (titanic_dataset["sex"] == 'female') & (titanic_dataset['survived'] == 1) & (
                                               titanic_dataset['pclass'] == 1)]))
    female_survived.append(len(titanic_dataset[
                                   (titanic_dataset["sex"] == 'female') & (titanic_dataset['survived'] == 1) & (
                                               titanic_dataset['pclass'] == 2)]))
    female_survived.append(len(titanic_dataset[
                                   (titanic_dataset["sex"] == 'female') & (titanic_dataset['survived'] == 1) & (
                                               titanic_dataset['pclass'] == 3)]))

    label = ['1st', '2nd', '3rd']

    x = np.arange(len(label))
    width = 0.35

    fig, axis = plot.subplots()
    rects1 = axis.bar(x - width / 2, male_survived, width, label='Male')
    rects2 = axis.bar(x + width / 2, female_survived, width, label='Female')

    axis.set_ylabel('Number of Persons')
    axis.set_xlabel('Class')
    axis.set_title('Survivors Group by Gender and Class')
    axis.set_xticks(x)
    axis.set_xticklabels(label)
    axis.legend()

    axis.bar_label(rects1, padding=3)
    axis.bar_label(rects2, padding=3)

    fig.tight_layout()

    plot.savefig('grouped_bar_chart_gender_vs_survivor_of_Pclass.svg')
    plot.show()


def hist_survivors_by_age():
    survived = np.array(
        titanic_dataset[(titanic_dataset['survived'] == 1) & (titanic_dataset['age'] != 0)]["age"].tolist())
    unsurvived = np.array(
        titanic_dataset[(titanic_dataset['survived'] == 0) & (titanic_dataset['age'] != 0)]["age"].tolist())
    plot.hist([survived, unsurvived], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80], edgecolor="black",
              label=['Survived', 'Unsurvived'])

    plot.title("Survivors by Age Group")
    plot.xlabel('Age Group')
    plot.ylabel('Number of Survivors')
    plot.legend()
    plot.savefig('hist_survivors_by_age.svg')
    plot.show()


def box_vs_violin():
    fare_of_pclass = [
        titanic_dataset[(titanic_dataset['pclass'] == 1) & (titanic_dataset['fare'] != 0)]['fare'].tolist(),
        titanic_dataset[(titanic_dataset['pclass'] == 2) & (titanic_dataset['fare'] != 0)]['fare'].tolist(),
        titanic_dataset[(titanic_dataset['pclass'] == 3) & (titanic_dataset['fare'] != 0)]['fare'].tolist()]
    figure, axis = plot.subplots(nrows=1, ncols=2, figsize=(10, 5))
    plot.suptitle("Fare of Each Class")
    axis[0].set_ylabel('Fare in $')
    axis[0].set_xlabel('Class')
    axis[1].set_ylabel('Fare in $')
    axis[1].set_xlabel('Class')

    axis[0].violinplot(fare_of_pclass, showmeans=False, showmedians=True)
    axis[1].boxplot(fare_of_pclass)
    axis[0].set_title('Violin Plot')
    axis[1].set_title('Box Plot')

    for item in axis:
        item.yaxis.grid(True)
        item.set_xticks([y + 1 for y in range(len(fare_of_pclass))])

    plot.setp(axis, xticks=[y + 1 for y in range(len(fare_of_pclass))],
              xticklabels=['1st', '2nd', '3rd'])
    plot.savefig('box_vs_violin.svg')
    plot.show()


if __name__ == '__main__':
    # stacked_bar_plot_gender_vs_survivor()
    # stacked_bar_plot_survivor_vs_class()
    grouped_bar_chart_gender_vs_survivor_of_Pclass()
    # hist_survivors_by_age()
    # box_vs_violin()
