from flask import Flask, request, make_response,send_file
from stemming.porter2 import stem
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from io import BytesIO
import time, zipfile
from flasgger import Swagger
import os




app = Flask(__name__)
swagger = Swagger(app)
port = int(os.environ.get("PORT", 5000))

def cleanse_text(text):
    if text:
        # fix whitespaces
        clean = text.split()
        # Stemming
        red_text = [stem(word) for word in clean]
        return " ".join(red_text)

    else:
        return text

# A welcome message to test our server
@app.route('/')
def index():
    return """<h1>Welcome to our text clustering webapp !!</h1> 
    <p>go to <a href="/apidocs">app page<a> to access the app<p>"""

@app.route("/cluster",methods=["POST"])
def cluster():
    """
    Example file endpoint returning a prediction of iris
    This is using docstring for specifications
    ---
    parameters:
      - name: dataset
        in: formData
        type: file
        required: true
      - name: col
        in: query
        type: string
        required: false
      - name: no_of_clusters
        in: query
        type: number
        required: false
    responses:
      200:
        description: OK
    """
    data=pd.read_csv(request.files.get("dataset"))
    unstructured = "text"
    if "col" in request.args:
        unstructured = request.args.get("col")
    no_of_clusters = 2
    if "no_of_clusters" in request.args:
        no_of_clusters = int(request.args.get("no_of_clusters"))
    data = data.fillna("NULL")
    data["clean_sum"] = data[unstructured].apply(cleanse_text)
    vectorizer = CountVectorizer(analyzer="word",
                                stop_words="english")
    counts = vectorizer.fit_transform(data["clean_sum"])

    kmeans = KMeans(n_clusters = no_of_clusters)
    data["Cluster_num"] = kmeans.fit_predict(counts)
    data.drop(["clean_sum"],axis=1,inplace=True)

    output = BytesIO()
    writer=pd.ExcelWriter(output,engine = "xlsxwriter")
    data.to_excel(writer,sheet_name="Clusters",encoding="utf-8",index=False)

    # find cluster centroids
    clusters = []
    for i in range(np.shape(kmeans.cluster_centers_)[0]):
        data_cluster = pd.concat([pd.Series(vectorizer.get_feature_names()),
                                  pd.DataFrame(kmeans.cluster_centers_[i])],
                                  axis= 1)
        data_cluster.columns = ["keywords","weights"]
        data_cluster=data_cluster.sort_values(by=["weights"],ascending=False)
        data_clust = data_cluster.head(n=10)["keywords"].tolist()
        clusters.append(data_clust)
    pd.DataFrame(clusters).to_excel(writer,sheet_name="Top_Keywords",encoding="utf-8")

    #Pivot
    data_pivot=data.groupby(["Cluster_num"],as_index=False).size()
    data_pivot.name ="size"
    data_pivot = data_pivot.reset_index(drop=True)
    data_pivot.to_excel(writer,sheet_name="Cluster_Report",encoding="utf-8")
    # Insert chart
    workbook = writer.book
    worksheet=writer.sheets["Cluster_Report"]
    chart=workbook.add_chart({"type":"column"})
    chart.add_series({
        "values":"=Cluster_Report!$B$2:$B"+str(no_of_clusters+1)
    })
    worksheet.insert_chart("D2",chart)

    writer.save()

    memory_file=BytesIO()
    with zipfile.ZipFile(memory_file,"w") as zf:
        names=["cluster_output.xlsx"]
        files=[output]
        for i in range(len(files)):
            data=zipfile.ZipInfo(names[i])
            data.date_time = time.localtime(time.time())
            data.compress_type=zipfile.ZIP_DEFLATED
            zf.writestr(data,files[i].getvalue())
    memory_file.seek(0)
    response=make_response(send_file(memory_file,attachment_filename="cluster_output.zip",
            as_attachment=True))
    response.headers["Content-Disposition"] = "attachment;filename=cluster_output.zip"
    response.headers["Access-Control-Allow-Origin"] = "*"

    return response
    

if __name__ == '__main__':
    app.run(threaded=True, port=5000)

