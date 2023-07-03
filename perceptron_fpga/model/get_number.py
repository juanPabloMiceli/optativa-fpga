import sys

'''
Use: python get_number.py <array_variable_name> <elements_variable_name> <numbers list>

Prints the following lines ready to copy and paste to a VHDL file.

<elements_variable_name>_0 <= <number_0 in 64 bits>; -- <number_0>
.
.
.
<elements_variable_name>_k <= <number_k in 64 bits>; -- <number_k>

<array_variable_name>(0) <= <elements_variable_name>_0;
.
.
.
<array_variable_name>(k) <= <elements_variable_name>_k;
'''
def main():
    array_variable_name = sys.argv[1]
    variable_name = sys.argv[2]
    output_str = ""

    for i in range(3, len(sys.argv)):
        original_number = sys.argv[i]
        number = round(float(original_number) * (2**32))
        if number < 0:
            binary_str = negative_str(number)
        else:
            binary_str = positive_str(number)
        number_with_underscores = add_underscores(binary_str)
        output_str += f"{variable_name}_{i-3} <= b\"{number_with_underscores}\"; -- {original_number}\n"

    output_str += "\n"
    for i in range(3, len(sys.argv)):
        output_str += f"{array_variable_name}({i-3}) <= {variable_name}_{i-3};\n"
    print(output_str)


def negative_str(input):
    return bin(input & 0xFFFF_FFFF_FFFF_FFFF)[2:]

def positive_str(input):
    return f"{input:064b}"

def add_underscores(my_str, group=16, char='_'):
    return char.join(my_str[i:i+group] for i in range(0, len(my_str), group))


def get_number_line(variable_name, index, original_number):
    output_str = ""
    shifted_number = round(float(original_number) * (2**32))
    if float(original_number) < 0:
        binary_number = negative_str(shifted_number)
    else:
        binary_number = positive_str(shifted_number)
    binary_number = add_underscores(binary_number)
    return f"\t{variable_name}({index}) := b\"{binary_number}\"; -- {original_number}"


if __name__ == '__main__':
    main()
