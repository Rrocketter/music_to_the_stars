from flask import Flask, request, jsonify

app = Flask(__name__)

# Define a sample function
def my_function(data):
    # Process data and return a result
    result = {"message": f"Hello, {data['name']}!"}
    return result


# Define a route to handle requests
@app.route('/run', methods=['POST'])
def run_function():
    # Get JSON data from the request
    data = request.get_json()
    # Call your function with the data
    result = my_function(data)
    # Return the result as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
