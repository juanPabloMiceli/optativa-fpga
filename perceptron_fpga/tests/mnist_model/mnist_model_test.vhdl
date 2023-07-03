library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.perceptron_package.perceptron_input;
use work.mnist_weights.GetWeights;
use work.mnist_inputs_9.GetInputs;


entity mnist_model_test IS
end mnist_model_test;

architecture behavior of mnist_model_test is 
    constant MATRIX_COLUMNS : natural := 3; -- Input + Middle + Output
    constant ROWS_PER_COLUMN : natural := 100;
    constant MATRIX_INPUTS : natural := 64;
    constant MATRIX_OUTPUTS : natural := 10;
    constant INPUT_WEIGHTS: natural := ROWS_PER_COLUMN * MATRIX_INPUTS;
    constant MIDDLE_WEIGHTS: natural := (MATRIX_COLUMNS-3) * (ROWS_PER_COLUMN * ROWS_PER_COLUMN);
    constant OUTPUT_WEIGHTS: natural := ROWS_PER_COLUMN * MATRIX_OUTPUTS;
    constant TOTAL_WEIGHTS: natural := INPUT_WEIGHTS + MIDDLE_WEIGHTS + OUTPUT_WEIGHTS;
	-- Inputs
	signal pesos_i : perceptron_input(TOTAL_WEIGHTS-1 downto 0);

	signal inputs_i : perceptron_input(MATRIX_INPUTS-1 downto 0);
	signal input_0 : std_logic_vector(63 downto 0);
	signal input_1 : std_logic_vector(63 downto 0);
	signal input_2 : std_logic_vector(63 downto 0);
	signal input_3 : std_logic_vector(63 downto 0);
	signal input_4 : std_logic_vector(63 downto 0);
	signal input_5 : std_logic_vector(63 downto 0);
	signal input_6 : std_logic_vector(63 downto 0);
	signal input_7 : std_logic_vector(63 downto 0);
	signal input_8 : std_logic_vector(63 downto 0);
	signal input_9 : std_logic_vector(63 downto 0);
	signal input_10 : std_logic_vector(63 downto 0);
	signal input_11 : std_logic_vector(63 downto 0);
	signal input_12 : std_logic_vector(63 downto 0);
	signal input_13 : std_logic_vector(63 downto 0);
	signal input_14 : std_logic_vector(63 downto 0);
	signal input_15 : std_logic_vector(63 downto 0);
	signal input_16 : std_logic_vector(63 downto 0);
	signal input_17 : std_logic_vector(63 downto 0);
	signal input_18 : std_logic_vector(63 downto 0);
	signal input_19 : std_logic_vector(63 downto 0);
	signal input_20 : std_logic_vector(63 downto 0);
	signal input_21 : std_logic_vector(63 downto 0);
	signal input_22 : std_logic_vector(63 downto 0);
	signal input_23 : std_logic_vector(63 downto 0);
	signal input_24 : std_logic_vector(63 downto 0);
	signal input_25 : std_logic_vector(63 downto 0);
	signal input_26 : std_logic_vector(63 downto 0);
	signal input_27 : std_logic_vector(63 downto 0);
	signal input_28 : std_logic_vector(63 downto 0);
	signal input_29 : std_logic_vector(63 downto 0);
	signal input_30 : std_logic_vector(63 downto 0);
	signal input_31 : std_logic_vector(63 downto 0);
	signal input_32 : std_logic_vector(63 downto 0);
	signal input_33 : std_logic_vector(63 downto 0);
	signal input_34 : std_logic_vector(63 downto 0);
	signal input_35 : std_logic_vector(63 downto 0);
	signal input_36 : std_logic_vector(63 downto 0);
	signal input_37 : std_logic_vector(63 downto 0);
	signal input_38 : std_logic_vector(63 downto 0);
	signal input_39 : std_logic_vector(63 downto 0);
	signal input_40 : std_logic_vector(63 downto 0);
	signal input_41 : std_logic_vector(63 downto 0);
	signal input_42 : std_logic_vector(63 downto 0);
	signal input_43 : std_logic_vector(63 downto 0);
	signal input_44 : std_logic_vector(63 downto 0);
	signal input_45 : std_logic_vector(63 downto 0);
	signal input_46 : std_logic_vector(63 downto 0);
	signal input_47 : std_logic_vector(63 downto 0);
	signal input_48 : std_logic_vector(63 downto 0);
	signal input_49 : std_logic_vector(63 downto 0);
	signal input_50 : std_logic_vector(63 downto 0);
	signal input_51 : std_logic_vector(63 downto 0);
	signal input_52 : std_logic_vector(63 downto 0);
	signal input_53 : std_logic_vector(63 downto 0);
	signal input_54 : std_logic_vector(63 downto 0);
	signal input_55 : std_logic_vector(63 downto 0);
	signal input_56 : std_logic_vector(63 downto 0);
	signal input_57 : std_logic_vector(63 downto 0);
	signal input_58 : std_logic_vector(63 downto 0);
	signal input_59 : std_logic_vector(63 downto 0);
	signal input_60 : std_logic_vector(63 downto 0);
	signal input_61 : std_logic_vector(63 downto 0);
	signal input_62 : std_logic_vector(63 downto 0);
	signal input_63 : std_logic_vector(63 downto 0);

	signal clock_i : std_logic := '1';
    signal clr_i : std_logic := '1';
	signal enable_i : std_logic := '1';

	-- Ouputs
    signal bcd_out : std_logic_vector(3 downto 0);

	-- Clock
	constant clock_period : time := 10 ms;

begin
	clock_process : process
	begin
		clock_i <= '0';
		wait for clock_period/2;
		clock_i <= '1';
		wait for clock_period/2;
	end process;

    pesos_i <= GetWeights(0);
    inputs_i <= GetInputs(0);
    input_0 <= inputs_i(0);
    input_1 <= inputs_i(1);
    input_2 <= inputs_i(2);
    input_3 <= inputs_i(3);
    input_4 <= inputs_i(4);
    input_5 <= inputs_i(5);
    input_6 <= inputs_i(6);
    input_7 <= inputs_i(7);
    input_8 <= inputs_i(8);
    input_9 <= inputs_i(9);
    input_10 <= inputs_i(10);
    input_11 <= inputs_i(11);
    input_12 <= inputs_i(12);
    input_13 <= inputs_i(13);
    input_14 <= inputs_i(14);
    input_15 <= inputs_i(15);
    input_16 <= inputs_i(16);
    input_17 <= inputs_i(17);
    input_18 <= inputs_i(18);
    input_19 <= inputs_i(19);
    input_20 <= inputs_i(20);
    input_21 <= inputs_i(21);
    input_22 <= inputs_i(22);
    input_23 <= inputs_i(23);
    input_24 <= inputs_i(24);
    input_25 <= inputs_i(25);
    input_26 <= inputs_i(26);
    input_27 <= inputs_i(27);
    input_28 <= inputs_i(28);
    input_29 <= inputs_i(29);
    input_30 <= inputs_i(30);
    input_31 <= inputs_i(31);
    input_32 <= inputs_i(32);
    input_33 <= inputs_i(33);
    input_34 <= inputs_i(34);
    input_35 <= inputs_i(35);
    input_36 <= inputs_i(36);
    input_37 <= inputs_i(37);
    input_38 <= inputs_i(38);
    input_39 <= inputs_i(39);
    input_40 <= inputs_i(40);
    input_41 <= inputs_i(41);
    input_42 <= inputs_i(42);
    input_43 <= inputs_i(43);
    input_44 <= inputs_i(44);
    input_45 <= inputs_i(45);
    input_46 <= inputs_i(46);
    input_47 <= inputs_i(47);
    input_48 <= inputs_i(48);
    input_49 <= inputs_i(49);
    input_50 <= inputs_i(50);
    input_51 <= inputs_i(51);
    input_52 <= inputs_i(52);
    input_53 <= inputs_i(53);
    input_54 <= inputs_i(54);
    input_55 <= inputs_i(55);
    input_56 <= inputs_i(56);
    input_57 <= inputs_i(57);
    input_58 <= inputs_i(58);
    input_59 <= inputs_i(59);
    input_60 <= inputs_i(60);
    input_61 <= inputs_i(61);
    input_62 <= inputs_i(62);
    input_63 <= inputs_i(63);

	clr_i <= '1', '0' after 200 ms;
	enable_i <= '0', '1' after 300 ms;

	uut1 : entity work.mnist_model
	port map (clock_i => clock_i,
		  clr_i => clr_i,
		  enable_i => enable_i,
          pesos_i => pesos_i,
		  inputs_i => inputs_i,
		  bcd_digit_o => bcd_out
		  );
end architecture;
