import sys
from get_number import get_number_line

'''
Trade offer:
    You give me: 
        * the name of the package you want to create.
        * A file with 1 weight per line
        * An output file
    You receive:
        * The necessary code for getting the weights from the desired package written in the ouput file

Example: python generate_weights_package.py <package_name> <weights_file> <output_file>
'''
def main():
    if(len(sys.argv) < 4):
        print(f'Missing parameters. Proper usage: \n\tpython {str(sys.argv[0])} <package_name> <weights_file> <output_file>')
        return
    package_name = sys.argv[1]
    weights_file = sys.argv[2]
    output_file = sys.argv[3]
    if len(sys.argv) >= 5:
        generated_function = sys.argv[4]
    else:
        generated_function = 'GetWeights'

    weights = []
    with open(weights_file) as file:
        for weight in file:
            if weight.isspace():
                continue
            weights.append(weight)

    write_weights_to_file_from_list(weights, package_name, output_file, generated_function)

def write_weights_to_file_from_list(weights, package_name, output_file, generated_function):
    total_weights = len(weights)
    weights_str = ""
    for index, weight in enumerate(weights):
        weights_str += get_number_line("pesos_i", index, weight) + "\n"


    write_weights_to_file(weights_str, total_weights, package_name, output_file, generated_function)


def write_weights_to_file(weights_str, total_weights, package_name, output_file, generated_function):
    file_str = (
f'''
-- ARCHIVO AUTOGENERADO CON {str(sys.argv[0])}

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.perceptron_package.perceptron_input;

package {package_name} is
    function {generated_function}(Dummy: natural)
    return perceptron_input;
end package {package_name};

package body {package_name} is
    function {generated_function}(Dummy: natural) return perceptron_input is
        variable pesos_i : perceptron_input({total_weights-1} downto 0);
    begin
{weights_str}
    return pesos_i;
    end function;
end package body {package_name};
    ''')
    vhdl_file = open(f"{output_file}", "w")
    vhdl_file.write(file_str)
    vhdl_file.close()



if __name__ == '__main__':
    main()
