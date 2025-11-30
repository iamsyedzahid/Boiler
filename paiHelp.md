INCLUDE Irvine32.inc

.data
myString BYTE "hello world", 0
buffer   BYTE 50 DUP(?)
strLength DWORD ?

.code
main PROC
    ; 1. Display a String (WriteString)
    ; Parameter: Address of myString (EDX)
    mov edx, ADDR myString 
    call WriteString
    call CrLf

    ; 2. Calculate Length (Str_length)
    ; INVOKE Str_length, Address of string
    INVOKE Str_length, ADDR myString
    mov strLength, eax ; EAX now holds 11

    ; 3. Convert to Uppercase (Str_ucase)
    ; INVOKE Str_ucase, Address of string (modifies myString)
    INVOKE Str_ucase, ADDR myString

    ; 4. Copy and Display the converted string
    INVOKE Str_copy, ADDR myString, ADDR buffer
    
    mov edx, ADDR buffer
    call WriteString   ; Output: HELLO WORLD
    call CrLf
    
    exit
main ENDP
END main
<img width="743" height="617" alt="image" src="https://github.com/user-attachments/assets/0080f559-2ccb-4ed9-8bf4-bf1f26b2206d" />

