import { Container } from "react-bootstrap";
import DocCard from "./DocCard";

function SearchResults(props) {

    const docArray = props.data;
    console.log("from SearchResults", docArray);

    return (
        <Container>
            <ol>
            {
                docArray.map((value, index) => {
                    return (<DocCard key={index} meta={value}/>)
                })
            }
            </ol>
        </Container>
    )
}

export default SearchResults;