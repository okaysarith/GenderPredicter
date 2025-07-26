from flask import Flask
from app import views
app = Flask(__name__,template_folder='E:/comp vision flask app/MY main/Flask_app/template')#here normally the template was not detected so path was added


app.add_url_rule(rule='/',endpoint='home',view_func=views.index)
app.add_url_rule(rule='/app',endpoint='app',view_func=views.app)
app.add_url_rule(rule='/app/gender/',
                 endpoint='gender',
                 view_func=views.genderapp,methods = ['GET','POST'])

if __name__ == "__main__":
    app.run(debug=True)