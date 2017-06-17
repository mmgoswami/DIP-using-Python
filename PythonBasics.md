
# Scientific Computing using Python

## Mukesh M Goswami

![DDUlogo](images/ddulogo.png) 	 
## Dharmsinh Desai University-Nadiad
#### Faculty of Technology
#### Department of IT
<!-- page_number:true -->

---
# Why Python??
* Free and Open source 
* Platform Independent/ Interpreted (No compilation required!!!) 
* Rich Package Library for Scientific and Engineering Computation
* Easy to learn (English Like Syntax)
* Rich Community Support
* Greate Integration with C\C++ (hence Faster!!!)
* Functional and Object Oriented
* Resources
-- Download: [http://www.python.org](http://www.python.org)			
-- Document: [http://www.python.org/docs](http://www.python.org/docs)
-- Free e-Books: [http://www.diveintopython.org](http://www.diveintopython.org) 

---

# Python -"Hello World!"

```
* Interactive window
C:\User>ipython
Python 2.5 (r25:51908, May 5 2017, 16:14:04)
[GCC 4.1.2 20061115 (prerelease) (SUSE Linux)] on linux2
Type "help", "copyright", "credits" or "license" for more information

In[1]: print("Hello World!")
Out[1]:Hello Word!
In[2]: 3*(4+2)
Out[2]:18
In[3]: x,y= 2,3 
```

```
* Running Scripts
#module: pyscript.py
#author: M M Goswami
def power(n,p):
	return(n**P)
	
print(power(2,3))
C:\Users> python pyscript.py
32
```
---


# White Space Matters
* Case sensitive, New Statement on New Line 
* `:`gives begining of the block, All statements within block are at same level
* `#` single line comment and `'''Multiline Comment'''` 
>```
>#module: pyscript.py
>#author: M M Goswami
>def power(n,p):
>	statement1
>	return(n**P)
>
>print(power(2,3))
* Naming Convection same as C\C++
* Does not support **function overloading** 
 
 
---

# Basic Data Types
Four Basic Data Types  
```md 
int, float, bool, string
``` 
*`int` - Complete number 
*`float` - Fractional number
*`bool`- _True_ or _Flase_
*`string` - characters/text
>Python is a dynamically typed langauge. _Type is automatically determined based on the values assigned to variables_  

- `name = 'Mukesh Goswami'`
- `area= 10.0`
- `units = 20`

> Once assigned a _type can not be automatically converted_

---

# Basic Operations 
>Mathematical Operations (works with `int` and `float`)
>`+, -, *, /` and `//, %, **` 

>Comparision Operator (works with all datatype, outcome is _always bool_)
> > `<, <=, >, >=, ==, !=` 

>Logical Operations (works with `bool`)
>`and, or, not` and `is` 

>String Operations (works with `string`)
>`+` concatations


---


# Strings
* Sequence (**_list_**) of charcaters
* Immutable Datatype

>```
>>> name = 'Mukesh Goswami'
>>> print(name)
>>> 'Mukesh Goswami'  
>>> text = "Oscam's Razor"
>>> print(text)
>>> "Oscam's Razor"
>>> para= '''This is a   
>>> 	multiline test'''
>>> print(para)
>>> "This is a\n multiline test"
>>> greetings = "Hello" + 'World'
>>> print(greetings)
>>> "HelloWorld"
>>> print(greetings[:4])
>>> "Hell"
>>> print(greetings[0]) 
>>> "H" #Does not diff btn Character and String
>>> greetings[0]="H" #Error

----


# Python Collection Types:
* **List** (_mutable_), **Dictionary**(_mutable_), **Tuple**(_immutable_), **Strings**(_immutable_)
* Index starts from _**zero**_
* List, Dictionary, and Tuples: Generic and Nested
>``` 
>>> l=[1,'abc',True, 24.4]
>>> l[1]
>>> 'abc'
>>> l[-2]
>>> True
>>> s = "Mukesh Goswami"
>>> s[1]
>>> 'u'
>>> t=(1, "abc", Ture, (2,3), 5.6)
>>> t[0]
>>> 1
>>> t[-1]
>>> 5.6
>>> t[-2]
>>> (2,3)

---

# Slicing the Collections
>```
>>> l=[1,'abc',True, 24.4]
>>> l[:2] #top slice
>>> [1,'abc']
>>> s = "Mukesh Goswami"
>>> s[1:] #bottom slice
>>> "ukesh Goswami"
>>> t=(1, "abc", Ture, (2,3), 5.6)
>>> t[2:4] 
>>> (Ture, (2,3)) 
>>> s[:] #full sclice
>>> "Mukesh Goswami"

* Slice always returns a _**new copy**_ of collection
* Collection Method
	* `len(s), del(t[2]), upper(s), lower(s), max(t), min(t)`

---
# Collection Operators
* `in`- Membership Operator
>```
>>>>t=[1,2,3,4]
>>>>3 in t
>>>>True
>>>>4 not in t
>>>>False
>>>>'uke' in 'Mukesh'
>>>>True
>```
* '+' - Concatanation 
>```
>>>>print(t+t)
>>>>[1,2,3,4,1,2,3,4]
* '*'- Repeate
>```
>>>>print(t*2)
>>>>[1,2,3,4,1,2,3,4]
>```

---

# Immutability (List vs. Tuples)

>```
>>>>t=(1, "abc", Ture, (2,3), 5.6)
>>>>t[0]=10
>""""Traceback (most recent call last):
>File "<pyshell#75>", line 1, in -toplevel-
>t[0] = 10
>TypeError: object doesn't support item assignment""""
>```
>```
>>>l1=[1, "abc", Ture, (2,3), 5.6]
>>>l1[0]=10
>>>print(l1)
>>>[10, "abc", Ture, (2,3), 5.6]
>>>l2=l1
>>>l3=l1[:]
>>>l2[-1]=20.5
>>>print(l1)
>>>[10, "abc", Ture, (2,3), 20.5]
>>>print(l3)
>>>[10, "abc", Ture, (2,3), 5.6]
>```

---

# List Functions

>```
>>>l=[1,2,3,4]
>>>l=l+l
>>>l.extend(l)
>>>l.append(1)
>>>l.sort()
>>>l.reverse()
>>>l.count(1)
>>>l.index(1)
>>>l.remove(1)
>```

---

# Type Conversion

>```
>>>>t= tuple(l) #List to Tuple
>>>>l=list(t) #Tuple to List
>>>>i=str(10) #num to str
>>>>i=int('10') #str to num
>>>>f=float(10) #int to float
>>>>i=int(22.2) #float to int
>>>>b=bool(1) #True
>>>>b=bool(-1) #True
>>>>b=bool(0) #False
>```
---

# Disctionary

* Maps set of keys with set of values, keys are unique and string type, values are any type
* **create, view, delete, update, list**
>```
>>>>d={} #Empty Dictionary
>>>>d={'user':'mmg', 'pass':12345}
>>>>d['user'] #get value by key
>>>>'mmg'
>>>>d['id']=007 #add new key-value pair
>>>>d['user']='xyz' #update value for key 'user'
>>>>del(d['user']) #delete entry given by key
>>>>d.keys() #list all keys (sorted)	
>>>>d.values() #list all values (sorted by key)
>>>>d.items() #list item tuples (key,value)
>>>>d.clear() #Remove all elements
>```


---

# Control and Loops
```
if (x == 3):								
  print("X equals 3.")
elif(x == 2):
  print("X equals 2.")
else:
  print("X equals something else.")
print("This is outside the â€˜ifâ€™.")
```
```
while(x < 10):
  if (x > 7) and (x < 10):
    x += 2
    continue
  x = x + 1
print "Outside of the loop."
```
```
for x in range(10):
  if (x > 7) and (x < 10):
    x += 2
 	continue
  x = x + 1
print("Outside of the loop.")
```
---

# Function and Packages

* **Function** : A piece of _**reusable code**_
* **Input**: takes zero or more variables as input 
* **Output**: always return some value (**None** by default)
* **does not support overloading, assgin function to variables, pass function as an argument to other function, return function from other function, always pass by values**   
>```
>def functionname(a,b,c=10):
>	statement1
>	statement2
>	return ((a+b)/c)
>functionname(10,30,40)
>functionname(10,30)
>functionname(10,c=30,b=40)
>functionname(b=20,a=20)
>``` 

---

# Functions (contd.)

>```
>def sqr(x):
>	return (x**2)
>def lapply(f,lst)
>	result=[]
>	for ele in lst:
>		result.append(f(ele))
>	return result
>def lapply2(lst, f=lambda z:z**3)
>	result=[]
>	for ele in lst:
>		result.append(f(ele))
>	return result 			

---

# Packages 

* A package (module) is collection of function 
* divides the namespace
>```
>#your.py
>def calculation():
>	bla 
>	bla
>#my.py
>def calculation():
>	more
>	bla
>	bla	
>>>calculation() #which one???
>>>my.calculation() #calculation in my package
>>>your.calculation() #calculation in your package
>```
* load packages `import packagename`
* load function 'from packagename import functionname'
---

# Batteries of Packages 
* `numpy` - matlab like numerical computation, linear alg. 
* `scipy`- algorithm for optimization, linear alg., calculus etc.
* `matplotlib`- matlab like ploting 
* `sklearn`- machine learning
* `pil` - basic image processing
* `skimage`- matlab like image process
* `opencv(cv2)`- computer vision and video processing
* `PyOpenGL`- 3D reconstruction
* `theano`- tensor graph based computing (with GPU support)
* `pandas`- Data analytics using R like DataFrame
* `pattern`- web analytics, twitter, fb 
* `nltk`- natural language processing

---

# Numpy Basics
>Want to Compute `bmi` of patients from `weight` and `height`
>```
>wght=[30, 40, 50, 66, 77]
>hght=[4, 4, 5, 5.5, 6]
>bmi=wght/hght**2
>"Error Pyhon does not know how to apply math operator 
>to list"

>Numpy array supports elementwise operations (just like _**Matlab**_)
```
>>>import numpy as np
>>>np_wght=np.array(wght)
>>>np_hght=np.array(hght)
>>>bmi=np_wght/np_hght**2

```
---
# 1D Numpy Array
* All elements are of the same type
```
>>>x= np.array(["x", True, 1])
>>>print(x)
>>>['x','True','1']
>>>x= np.array([20.2, True, 1])
>>>[20.2,1.,1.]
```
* Indexing is done just like `list`
* Operations are perfomed elementwise
* Also supports binary index
```
>>>bmi=np.array([20.2, 18.0, 21.2, 22.34,])
>>>print(bmi<21)
>>>[True, True, False, False]
>>>print(bmi[bmi<21])
>>>[20.2, 18.0]
```
---

# List vs Array Operator
```
>>> list1=[1,2,3]
>>> list2=[1,2,3]
>>> list3=list1+list2
>>> print(list3)
>>> [1,2,3,1,2,3]
>>> np1=np.array([1,2,3])
>>> np2=np.array([1,2,3])
>>> np3=np1+np2 
>>> print(np3)
>>> [2,4,6]
>>> print(np.array([True, 20, 1]) + np.array([1, 20, False]))
>>> [2,40,0]
>>> type(np1)
>>> numpy.ndarray

```
---

# 2D Numpy Array
* Created from _list of list_
```
>>>mat=array([[1,2,3],[4,5,6],[7,8,9]], dtype=int)
>>>print(mat)
>>>[[1,2,3],
    [4,5,6],
    [6,7,8]]
>>>print(mat.shape)
>>>(3,3)
>>>print(mat[0][1]) or print(mat[0,1])
>>>[2]
>>>print(mat[2]) or print(mat[2,:])
>>>[7,8,9]
>>>print(mat[:,1])
>>>[2,5,8]
>>>print(mat[1:2,0:1])
>>>[[4,5],[7,8]]

```
> In fact you can create an array of any dimension by nesting the _list_

---

# 2D Array Arithmetic 

```
>>>import numpy as np
>>>mat=np.array([[1,2,3],[4,5,6],[7,8,9]])
>>>convert=np.array([0,2,3])
>>>print(mat * convert)
>>>[[0, 4, 9],
	[0, 10, 18],
	[0, 16, 27]]
```

> Numpy always perform _**element wise arithmetic operation**_  

```
np.zeros((5,10), dtype=int), 
np.ones(3, dtype=float), 
np.transpose(), #or np.T
np.identity(4, dtype=int)
np.random.uniform(size=9).reshape((3,3))
```

---

# Numpy Basic Statistics

* `np1.mean(), np1.median(), np1.min(), np1.max(), `
* `np.corrcoef(np1,np2), np1.argmin(), np1.argmax()`
* `np1.std(), np1.round()`
* `np1.sum(), np1.sum(axes=0) np1.sum(axes=1)`
* `np.random.normal(1.75,0.20,5000)`
* `np.random.uniform(0,1,100)`
* `np.column_stack((np1, np2))`
* `np.sin(np1), np.cos(np1), np.exp(np1)`
* _**and many more....**_ 

---
# `Matplotlib` Basic Plot 

```
import matplotlib.pyplot as plt
import numpy as np

X=np.linspace(-np.pi,np.pi,250,endpoint=True)
C,S=np.cos(X),np.sin(X)
plt.title("Sinosudal Plot")
plt.xlabel("Angle in Radian")
plt.ylabel("Sine")
plt.plot(X,C,'r',X,S,'b') 
#Format string 'r','g','b': red green, blue 
#'o', 's','^': circle, square, triangle
#'-', '--','o-','s-': normal, deshed, cirle with normal etc. 
```
![Sine-Cosine Plot](images/plot1.png)    

---

# 'Matplotlib' Subplot
```
import matplotlib.pyplot as plt
import numpy as np

X=np.arange(10)
Y1=X
Y2=X**2
Y3=X**3
plt.figure(1) #Optional here
plt.subplot(221)
plt.title("Linear")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(X,Y1,'ro')
plt.subplot(222)
plt.title("Square")
plt.xlabel("X")
plt.ylabel("Y^2")
plt.plot(X,Y2,'bs')
```
---
# Other Plots

```
X= np.random.uniform(size=100)
Y= np.random.uniform(size=100)
plt.scatter(X,Y)
plt.show()
```

```
mu, sigma=100,15
data = mu+sigma*np.random.randn(100)

plt.hist(data,50,facecolor='g', edgecolor='k', alpha=0.75)
plt.text(100,7,r'$\mu=100,\sigma=15$')
plt.show()
```
```
def f(x,y):
    return (x**2+2*x*y+y**2)
x=np.linspace(-1,1,256)
y=np.linspace(-1,1,256)
X,Y= np.meshgrid(x,y)
plt.contourf(X,Y,f(X,Y),cmap='jet')
plt.contour(X,Y, f(X,Y), colors='black')
```
---

# `scipy` Basics



---
# `scipy` Library Reference

* `scipy.io`- `savemat` and `loadmat` 
* `scipy.linalg`- Linear Algebra `det, inv, svd`
* `scipy.stats`- Statistics and Random Numbers
* `scipy.fftpack`-Fourier Transforms `fft, ifft`
* `scipy.optimize`- Numerical Optimization
* `scipy.interpolate `- Interpolation
* `scipy.integrate`- Numerical Integration
* `scipy.signal` - Signal Processing
* `scipy.ndimage`- nd-image processing
* `scipy.special`- Bessels, Jacobian, Gamma Functions etc.
* `scipy.cluster` - Vector Quantization, K-means
* `scipy.constants`- Physical and Methematical Constant

---

# `skimage` Basics

```
from skimage import data, filter
from matplotlib import pyplot as plt

img = data.coins()
plt.subplot(221)
plt.title("Pesudo Color Image Coin")
plt.axis('off')
plt.imshow(img)
plt.subplot(222)
plt.title("Gray Image Coin")
plt.axis('off')
plt.imshow(img, cmap=plt.cm.gray)
plt.subplot(223)
plt.axis('off')
plt.title("Edge Map Coin")
plt.imshow(filter.sobel(img), cmap=plt.cm.gray)
plt.subplot(224)
plt.title("DDU Logo Color Image")
plt.axis('off')
plt.imshow(io.imread('ddulogo.png'))
```

---

# `skimage' Library Reference

* `skimage.io`- `imread`, `imwrite`, `ImageCollection('data/*')` and `imshow` 
* `skimage.filter`- Point and Neighborhood Processing, Edge Detection,
* `skimage.segmentation`-  Segmentation
* `skimage.color`- Color Image Processing
* `skimage.transform` - Geometric and other transforms
* `skimage.features`- Feature Extraction
* `skimage.expose`- Contract streaching, correction and hist. equilization
* `skimage.morphology`- Morphology Operations
* `skimage.restoration`- Denoising and Restroration
* `skimage.measure`- profiles, energy, moments, comparision (MSE), etc.