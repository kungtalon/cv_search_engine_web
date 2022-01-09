import ReactMarkdown from "react-markdown";
import {useState, useEffect} from "react";
import { Container, Card } from "react-bootstrap";
import rehypeRaw from 'rehype-raw';
import IntroHtml from './IntroContent';

function MarkdownIntro(){
    const readmePath = require("../markdown/introduction.md");
    const [markdown, setMarkdown] = useState('');

    useEffect(() => {
        fetch(readmePath)
        .then(response => response.text())
        .then(text => setMarkdown(text))
    })
    
    return (
        <Container>
            <div className="intro-markdown">
                <div className="jumbotron" align="center">
                    <h2>A Search Engine for Academic Computer Vision Papers</h2>
                </div>
                <div className="markdown-body">
                    <ReactMarkdown children={markdown} rehypePlugins={rehypeRaw}/>
                </div>
            </div>
        </Container>
        
    )
}

function TableOfContents(){

    return (
        <div id="intro-contents-table">
            <Card border="secondary" style={{width:"18rem"}}>
                <Card.Header>
                    <span style={{fontWeight:'bold'}}>Table of Contents</span>
                </Card.Header>
                <Card.Body>
                    <ul>
                        <li><a href="#1-introduction">1. Introduction</a></li>
                        <li><a href="#2-data">2. Data</a></li>
                        <li>
                            <a href="#3-methods">3. Methods</a>
                            <ul>
                                <li><a href="#31-word2vec">3.1 Word2Vec</a></li>
                                <li><a href="#32-preparing-the-features">3.2 Preparing the features</a></li>
                                <li><a href="#33-learn-to-rank-with-lightgbm">3.3 Learn-to-rank with LGBM</a></li>
                            </ul>
                        </li>
                        <li><a href="#4-results-and-discussions">4. Results and Discussions</a></li>
                        <li><a href="#5-whats-next">5. What's next</a></li>
                    </ul>
                </Card.Body>
            </Card>
        </div>
    )
}


function HTMLIntro(){
    
    return (
        <div className="intro-page">
            <TableOfContents/>
            <Container>
                <div className="intro-markdown">
                    {/* <div className="jumbotron" align="center">
                        <h2>A Search Engine for Academic Computer Vision Papers</h2>
                    </div> */}
                    <div className="markdown-body">
                        <span dangerouslySetInnerHTML={{__html: IntroHtml}}></span>
                    </div>
                </div>
            </Container>
        </div>
    )
}

export default HTMLIntro;