# Илюстрация на малко матричвна алгебра с # numpy. 
# Изчислява динамичен множител на диференчното уравнение от трети ред.
# y_t = 0.3 * y_{t-1} + 0.9 * y_{t-2} + 0.2 * y_{t-3}

import numpy as np
import matplotlib.pyplot as plt

# Линейната алгебра в NumPy се извършва направо върху numpy масивите
F = np.array([[0.3, 0.9, 0.2], 
              [1, 0, 0], 
              [0, 1, 0]]);


# numpy.linalg.eig изчислява собствени стойности и собствени вектори.
# Собствените вектори се връщат в двумерен масив, така че стойността 
# на вектора, който отговаря на n-тата собствена стойност, може да се 
# прочете от колоната egenvectors[:, n]
eigenvalues, eigenvectors = np.linalg.eig(F);

print("Eigenvalues of F:");
print(eigenvalues);

print("Eigenvectors of F:")
print(eigenvectors);

# Тази стъпка не е нужна, но я правим, за да докараме матрицата със 
# собствените вектори така, както я гласи Wolfram Alpha. 
# Ясно е, че собствените вектори, които отговарят на дадена 
# собствена стойност, са безкрайно много, като те формират векторно 
# подпространство, което може да има едно или повече измерения. 
# При представянето на измеренията, Wolfram Alpha избира векторите 
# така, че последният компонент е единица.
# Долните изчисления правят това по нескопосан начин (няма цикъл). 

eigenvectors[:,0] = eigenvectors[:,0]*(1/eigenvectors[-1,0]);
eigenvectors[:,1] = eigenvectors[:,1]*(1/eigenvectors[-1,1]);
eigenvectors[:,2] = eigenvectors[:,2]*(1/eigenvectors[-1,2]);

print("Wolfram-like representation of the eigenvectors:");
print(eigenvectors);

# В книгата на Хамилтън, матрицата множител за диагонализация е отбелязана с T
T = eigenvectors;
print("T:");
print(T);
TInv = np.linalg.inv(T);

print("T^{-1}:");
print(TInv);

# Коефициентите във формулата за динамичния множител
c0 = T[0,0]*TInv[0,0];
c1 = T[1,0]*TInv[0,1];
c2 = T[2,0]*TInv[0,2];
      
print ("c0:");      
print (c0)

print ("c1:");      
print (c1)

print ("c2:");      
print (c2)

# Коефициентите трябва да се сумират до 1
print ("c0+c1+c2:")
print(c0+c1+c2);

# Накрая динамичния множител

multipliers = [];
for i in range (1, 10):
    dynMul = c0*pow(eigenvalues[0],i)+c1*pow(eigenvalues[1],i)+c2*pow(eigenvalues[2],i);
    print("Dynamic multiplier t=:"+str(i));
    print(dynMul);
    multipliers.append(dynMul);
    
fig, ax = plt.subplots();
ax.plot(multipliers, label="dynamic multiplier");
ax.legend();
    
