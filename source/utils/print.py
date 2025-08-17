

def check_num(number):
    if number is None:
        return 'N/A'
    elif isinstance(number, float):
        number = round(number, 2)
        return str(number)
    else:       
        return str(number)