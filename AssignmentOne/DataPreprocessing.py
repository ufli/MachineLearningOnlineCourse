__author__ = 'ufli'

import pandas
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
print(data[:10])

print(data['Sex'].value_counts())

print(data['Survived'].value_counts())

print(100. * data['Survived'].value_counts()[1] / (data['Survived'].value_counts()[0] +
                                                  data['Survived'].value_counts()[1]))

print(data['Pclass'].value_counts())

print(100. * data['Pclass'].value_counts()[1] / (data['Pclass'].value_counts()[2] +
                                                  data['Pclass'].value_counts()[3] +
                                                 data['Pclass'].value_counts()[1]))

print(pandas.Series.mean(data['Age']))
print(pandas.Series.median(data['Age']))


print(data['SibSp'].corr(data['Parch']))

data_women = data[data['Sex'] == 'female']

names = data_women['Name']
# print(names[1])
# print(names[2])
# print(names[3])
# print(names[4])
# print(names[5])

print
count = 0
firstNames = {}
for name in names:
    # firstName = name.split('. ')[1].split(' ')[0]
    firstName = ''
    prefix = name.split(', ')[1]
    try:
        if 'Mrs' in prefix:
            firstName = prefix.split('(')[1].split(')')[0].split(' ')[0]
        elif 'Miss' in prefix:
            firstName = prefix[6:].split(' ')[0]
        elif 'Ms' in prefix:
            firstName = prefix[4:].split(' ')[0]
        else:
            firstName = prefix.split(' ')[0]
        # print firstName
        try:
            firstNames[firstName] += 1
        except:
            firstNames[firstName] = 1
    except:
        count += 1
        pass
        # print ';s;d;fkf ' + prefix

print
print(count)

max = 0
popularName = ''
for firstName in firstNames:
    if firstNames[firstName] > max:
        popularName = firstName
        max = firstNames[firstName]

print(popularName)
print(firstNames[popularName])

print
print(firstNames.keys())
