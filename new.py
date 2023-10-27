ss=''
s='PyTHoN-2020'
size=len(s)
for I in range(size):
    if s[I] >='M' and s[I] <='U':
         ss+= s[I-2]
    elif s[I].isdigit():
         ss+= str(int(s[I]) * 3)
    else:
        ss+=s[I].upper()
print(ss)