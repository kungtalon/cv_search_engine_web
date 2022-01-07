import ReactMarkdown from "react-markdown";
import {useState, useEffect} from "react";
import { Container } from "react-bootstrap";
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

function HTMLIntro(){
    
    return (
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
        
    )
}

export default HTMLIntro;