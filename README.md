<h1>Beer Predictions</h1>
    <h2>Overview</h2>
        <table>
            <tr>
                <td><i>Project Description</i>:</td>
                <td>
                    <p>With the intention of predicting a particular Beer type, a Neural Network was developed, trained and deployed.</p>
                </td>
            </tr>
            <tr>
                <td><i>Project Objectives</i>:</td>
                <td>
                    <p>
                        1. Design a Neural Network using <a href="https://pytorch.org/">PyTorch</a> library.
                        <br>
                        2. Deploy the model behind an API endpoing using <a href="https://fastapi.tiangolo.com/">FastAPI</a>.
                        <br>
                        3. Test, document, and publish.
                    </p>   
                </td>
            </tr>
        </table>              
    <h2>Endpoints</h2>
        <table>
            <tr>
                <td><code><a href="/">/</a></code></td>
                <td>The root of the directory, which displays all the key information and details about the project, including: Overview, Endpoints, Model Info and More Info.</td>
            </tr>
            <tr>
                <td><code><a href="/docs">/docs/</a></code></td>
                <td>All of the detailed documentation. Read here first.</td>
            </tr>
            <tr>
                <td><code><a href="/health">/health/</a></code></td>
                <td>Check to ensure that the app is healthy and ready to run.</td>
            </tr>
            <tr>
                <td><code><a href="/beer/type">/beer/type/</a></code></td>
                <td>Check to review the architecture of the model.</td>
            </tr>
            <tr>
                <td><code><a href="/beers/type">/beers/type/</a></code></td>
                <td>Predict single Beer type, based on set input criteria.</td>
            </tr>
            <tr>
                <td><code><a href="/model/architecture">/model/architecture/</a></code></td>
                <td>Predict multiple Beer types, based on set input criteria.</td>
            </tr>
        </table>
    <h2>Model Info</h2>
        <table>
            <tr>
                <td><i>Input Parameters</i>:</td>
                <td>
                    <table>
                        <tr>
                            <th>Param</th>
                            <th>Type</th>
                            <th>Validation</th>
                        </tr>
                        <tr>
                            <td><code>brewery_name</code></td>
                            <td><code>str</code></td>
                            <td>Must be a valid brewery name</td>
                        </tr>
                        <tr>
                            <td><code>review_aroma</code></td>
                            <td><code>float</code></td>
                            <td>Must be between <code>0</code> and <code>5</code></td>
                        </tr>
                        <tr>
                            <td><code>review_appearance</code></td>
                            <td><code>float</code></td>
                            <td>Must be between <code>0</code> and <code>5</code></td>
                        </tr>
                        <tr>
                            <td><code>review_palate</code></td>
                            <td><code>float</code></td>
                            <td>Must be between <code>0</code> and <code>5</code></td>
                        </tr>
                        <tr>
                            <td><code>review_taste</code></td>
                            <td><code>float</code></td>
                            <td>Must be between <code>0</code> and <code>5</code></td>
                        </tr>
                    </table>
                </td>
            </tr>
            <tr>
                <td><i>Output Format</i>:</td>
                <td>
                    <table>
                        <tr>
                            <th>Format</th>
                            <th>Reason</th>
                        </tr>
                        <tr>
                            <td><code>str</code></td>
                            <td>If the input params are all scalar, then model will result in a single <code>str</code> prediction.</td>
                        </tr>
                        <tr>
                            <td><code>list</code> of <code>str</code></td>
                            <td>If the input params are a list, then the model will result in a <code>list</code> of <code>str</code>, who's length is the same as the number of input params.</td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    <h2>More Info</h2>
        <table>
            <tr>
                <td><i>Author</i></td>
                <td><a href="https://www.linkedin.com/in/chrimaho/">Chris Mahoney</a></td>
            </tr>
            <tr>
                <td><i>Repo</i>:</td>
                <td><a href="https://github.com/chrimaho/BeerPrediction">BeerPrediction</a></td>
            </tr>
            <tr>
                <td><i>Version</i></td>
                <td><code>0.1.0</code></td>
            </tr>
            <tr>
                <td><i>Published</i></td>
                <td>7/Mar/2021</td>
            </tr>
        </table>              