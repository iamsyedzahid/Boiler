
I. LAB 01 - 03: Fundamentals, Data Types, and Basic Instructions
Category
Concept
Key Directives/Instructions
Foundations
Assembly Language & Debugging
Low-level language, Breakpoints, Stepping ($\text{F10}$), $\text{Registers}$/$\text{Memory}$ Analysis.
Data Types
Intrinsic Data Types
$\text{BYTE}$ (1 byte), $\text{WORD}$ (2 bytes), $\text{DWORD}$ (4 bytes).
Memory
Data Definition
DB, DW, DD (or BYTE, WORD, DWORD) to reserve memory.
Strings
String Initialization
Null-terminated strings (ending with $\text{00h}$).
Basic Arithmetic
Data Transfer
MOV (Move), ADD (Addition), SUB (Subtraction).

II. LAB 04 - 05: Operators, Addressing, and Flags
Category
Concept
Operators/Instructions
Core Instructions
Data Manipulation
INC, DEC (Increment/Decrement), XCHG (Exchange).
Data Extension
Size Mixing
MOVSX (Move with Sign Extend), MOVZX (Move with Zero Extend).
Memory Size
Array Definition
DUP (Duplicate) operator for reserving large memory blocks.
Constants
Symbolic Constants
EQU (Equate) directive for defining names for constants.
Status Flags
$\text{EFLAGS}$ Register
Zero Flag ($\text{ZF}$), Carry Flag ($\text{CF}$), Sign Flag ($\text{SF}$), Overflow Flag ($\text{OF}$).
Size Operators
Variable Information
OFFSET (Address), TYPE (Element Size), LENGTHOF (Element Count), SIZEOF (Total Bytes).
Addressing Modes
Memory Access
Direct-Offset, Indirect (using registers), Indexed Operands.
Advanced Addressing
Operand Size Control
PTR (specifies byte, word, or dword access regardless of declaration).
Indexing
Array Iteration
Scale Factors (1, 2, 4, 8) used in indexed operands (e.g., $\text{[ESI * 4]}$).

III. LAB 06 - 07: Control Flow, Jumps, and Boolean Logic
Category
Concept
Instructions/Procedures
Comparison
Flag Setting
CMP (Compare) performs subtraction internally, only affecting the $\text{EFLAGS}$.
Boolean Logic
Bitwise Operations
AND, OR, NOT, XOR.
Bit Checking
Non-destructive AND
TEST (like $\text{AND}$, but only sets $\text{EFLAGS}$).
Program Flow
Jumps
JMP (Unconditional Jump), Jxx (Conditional Jumps like $\text{JE}$, $\text{JG}$, $\text{JLE}$).
Loops
Iteration Control
LOOP (uses $\text{ECX}$), LOOPZ}$**, **LOOPNZ}$ (Conditional Loops).
High-Level Logic
Conditional Structures
Implementing IF-ELSE and WHILE loops using jumps and labels.
Irvine I/O
Console Interaction
WriteString, ReadInt, WriteDec, CrLf.

IV. LAB 08 & 10: Stack and Advanced Procedures
Category
Concept
Directives/Instructions
Runtime Stack
LIFO Structure
LIFO (Last-In, First-Out) data structure for temporary storage.
Stack Operations
Data Access
PUSH (store data onto stack), POP (retrieve data from stack).
Procedure Definition
Modularity
PROC and ENDP directives.
Procedure Control
Subroutine Flow
CALL (pushes return address), RET (returns from procedure).
Stack Frame
Structured Access
Setting up the frame: PUSH EBP / MOV EBP, ESP to access parameters (e.g., $\text{[EBP+8]}$).
Advanced Invocation
Simplified Calls
INVOKE macro (replaces multiple $\text{PUSH}$ instructions).
Pointer Passing
Address Operator
ADDR operator (used with $\text{INVOKE}$ to pass an address/pointer).
Prototypes
Compile-Time Check
PROTO directive (declares procedure interface for type-checking).

V. LAB 09: Integer Arithmetic
Category
Concept
Instructions
Shifts
Logical/Arithmetic
SHL / SAL (Shift Left - Multiply), SHR (Shift Right), SAR (Arithmetic Right - Signed Divide).
Multiplication
Unsigned/Signed
MUL (Unsigned), IMUL (Signed).
Division
Unsigned/Signed
DIV (Unsigned), IDIV (Signed).
Sign Extension
Division Prep
CBW ($\text{AL} \to \text{AX}$), CWD ($\text{AX} \to \text{DX:AX}$), CDQ ($\text{EAX} \to \text{EDX:EAX}$).
Extended Precision
Carry/Borrow
ADC (Add with Carry), SBB (Subtract with Borrow).

VI. LAB 11: String Handling and 2D Arrays
Category
Concept
Instructions/Procedures
String Primitives
Specialized Data Ops
MOVSB/W/D, CMPSB/W/D, SCASB/W/D, STOSB/W/D, LODSB/W/D.
Repetition
Loop Prefixes
REP, REPE/REPZ, REPNE/REPNZ.
Pointers/Control
Register Usage
$\text{ESI}$ (Source Index), $\text{EDI}$ (Destination Index), $\text{ECX}$ (Counter), Direction Flag ($\text{CLD}$/$\text{STD}$).
2D Array Storage
Row-Major Order
Elements stored row-by-row sequentially in memory.
Address Calculation
Formula
$\text{BaseAddress} + (\mathbf{\text{RowSize}} \times \mathbf{\text{RowIndex}}) + (\mathbf{\text{ColIndex}} \times \mathbf{\text{ElementSize}})$.


