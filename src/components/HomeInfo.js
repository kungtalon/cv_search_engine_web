import {Card, Button, Container} from "react-bootstrap";

function HomeInfo() {


    return (
        <Container>
            <div className="home-card">
                <Card bg="Light">
                    <Card.Header>
                        <div className="home-card-header">
                            <p></p>
                        </div>
                    </Card.Header>
                    <Card.Body>
                        <div className="home-info-markdown">
                            <Card.Title>A Brief Introduction</Card.Title>
                            <Card.Text>
                                <div className="home-info-text">
                                    <p>💻 A search engine based on LightGBM model which can search for computer vision papers.</p>
                                    <p>🚚 The meta data of target papers are crawled from thecvf.com and arxiv.org with Selenium.</p>
                                    <p>🔎 To search for papers, please type in keywords and choose whether to search in titles/abstracts/sections.</p>
                                    <p>❤ To learn more about how this search engine works, please click Intro in the navigation bar.</p>
                                    <p>Frontend: ReactJS, HTML/CSS | Backend: Scikit-Learn, NLTK, PyTerrier, Django REST Framework.</p>
                                    <p>Course Project of EECS 549 Information Retrieval 21'Fall @ University of Michigan</p>
                                    <p>Author: Zelong Jiang</p>
                                    <p>Email: zelong@umich.edu</p>
                                </div>
                            </Card.Text>
                        </div>
                    </Card.Body>
                </Card>
            </div>
        </Container>
    )
}

export default HomeInfo;