import pyvisa as visa
import time

# Create a VISA resource manager
rm = visa.ResourceManager()

# List all available resources
resources = rm.list_resources()
print("Available VISA resources:", resources)

# Connect to the HP 33120A (change the address if needed)
gen = rm.open_resource('GPIB0::28::INSTR')  # Replace with your correct address if different
print("Connected to:", gen.query("*IDN?"))

# Try to reset the function generator
try:
    gen.write("FUNC SQU; FREQ 420; VOLT 2.5")


except Exception as e:
    print(f"Error: {e}")
finally:
    # Close the connection
    gen.close()
    print("Connection closed.")
