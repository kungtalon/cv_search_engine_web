import { Container } from "react-bootstrap";

function Error(props) {

    const imgSrc = "Crying_Emoji_Icon.png";

    const errorText = (status) => {
        if (status === 500){
            return "Search API Not Working!";
        } else {
            return "No response from API... Please try again 1 min later...";
        }
    }

    return (
        <Container>
            <div className="error-page" align="center">
                <img src={imgSrc} alt="Uh Oh!" width="200"/>
                <div className="error-message">
                    {errorText(props.statusCode)}
                </div>
            </div>
        </Container>
        
    )
}

export default Error;