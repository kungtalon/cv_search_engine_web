import { useState } from "react"
import { Form, Row, Col, InputGroup, FormControl, Button, DropdownButton, Dropdown } from "react-bootstrap";

function SearchBox(props) {

  const [searchQuery, setSearchQuery] = useState("");
  const [searchTitle, setSearchTitle] = useState(true);
  const [searchAbstract, setSearchAbstract] = useState(true);
  const [searchSections, setSearchSections] = useState(true);
  const [isSearching, setIsSearching] = useState(false);

  const fields = ["Title", "Abstract", "Sections"];
  const checkBoxStates = [searchTitle, searchAbstract, searchSections]

  function handleQueryChange(event) {
    setSearchQuery(event.target.value);
    console.log(searchQuery);
  }

  function handleCheckBoxChange(event) {
    if (event.target.id === "checkBoxTitle") {
      setSearchTitle(!searchTitle);
    } else if (event.target.id === "checkBoxAbstract") {
      setSearchAbstract(!searchAbstract);
    } else {
      setSearchSections(!searchSections);
    }
  }

  function doSearch(event){
    event.preventDefault();
    if(searchQuery.length > 0 && !isSearching){
      setIsSearching(true);
      const data = [
        searchQuery,
        ...checkBoxStates
      ]
      props.onSearch(data, doneSearch);
    }
  }

  function doneSearch(){
    setIsSearching(false);
  }

  function SearchIcon() {
    const rotating = { 
      animation:'spin 2s linear infinite'
    }
    return isSearching?<i className="icon fa fa-spinner" style={rotating} alt="···"></i>:<i className="icon fa fa-search"></i>;
  }

  return (
    <div className="col-sm-6 col-md-6 pull-right">
      <form className="navbar-form" role="search">
      <InputGroup>
          <FormControl aria-label="Text input with dropdown button" placeholder="Search..." onChange={handleQueryChange}/>
          <div className="input-group-btn">
              <Button type="submit" onClick={doSearch}>
                <SearchIcon/>
              </Button>
          </div>
          <div className="checkboxes">
            <Row>
            {
              fields.map((value, index)=>{  return (
                <Col key={index}>
                  <Form.Check
                    id={"checkBox" + value}
                    type="checkbox"
                    className="mb-2"
                    label={value}
                    checked={checkBoxStates[index]}
                    onChange={handleCheckBoxChange}
                  />
                </Col>
              )})
            }
            </Row>
          </div>
      </InputGroup>
      </form>
    </div>

  );
}

export default SearchBox;
