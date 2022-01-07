import { Card } from "react-bootstrap";

function DocCard(props) {
    const meta = props.meta

    function ConferenceLabel() {
        var color = (meta.conference === 'ICCV' ? 'rgb(245, 186, 19)':'rgb(33, 150, 243)');

        return (
            <div className="doc-card-label" style={{background: color}}>{meta.conference} {meta.year}</div>
        )
    }

    return (
        <div className="doc-card">
            {/* <div className="doc-card-head">

            </div> */}
            <div className="doc-card-body">
                <div className="doc-title">
                    <div className="doc-title-text">{meta.title}</div>
                    <ConferenceLabel/>
                </div>
                <p className="doc-authors"><span style={{fontWeight:'bold'}}>Authors</span>: {meta.authors}</p>
                <p className="doc-abstract"><span style={{fontWeight:'bold'}}>Abstract</span>: {meta.abstract}</p>
                {meta.workshop !== '' && <p className="doc-workshop"><span style={{fontWeight:'bold'}}>Workshop</span>: {meta.workshop}</p>}
            </div>
        </div>
    );
}

export default DocCard;