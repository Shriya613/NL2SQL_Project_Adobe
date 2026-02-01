BEGIN TRANSACTION;
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Authors" VALUES(1,'Alice Johnson','alice@example.com');
INSERT INTO "Authors" VALUES(2,'Bob Brown','bob@example.com');
CREATE TABLE Journals(
    id INTEGER PRIMARY KEY,
    journal_name TEXT,
    ISSN TEXT UNIQUE
);
INSERT INTO "Journals" VALUES(1,'Journal of Information Technology','12345678901234567890');
INSERT INTO "Journals" VALUES(2,'International Journal of Computer Science','09876543210987654321');
CREATE TABLE Papers_Authors(
    paper_id INTEGER REFERENCES Research_Papers(id),
    author_id INTEGER REFERENCES Authors(id),
    PRIMARY KEY(paper_id,author_id)
);
INSERT INTO "Papers_Authors" VALUES(1,1);
INSERT INTO "Papers_Authors" VALUES(2,1);
CREATE TABLE Publications(
    id INTEGER PRIMARY KEY,
    paper_id INTEGER REFERENCES Research_Papers(id),
    journal_id INTEGER REFERENCES Journals(id),
    volume INT,
    issue INT,
    page_start INT,
    page_end INT
);
INSERT INTO "Publications" VALUES(1,1,1,10,1,1,10);
INSERT INTO "Publications" VALUES(2,2,2,5,2,250,259);
CREATE TABLE Research_Papers(
    id INTEGER PRIMARY KEY,
    title TEXT,
    abstract TEXT,
    publication_date DATE,
    citation_style TEXT 
);
INSERT INTO "Research_Papers" VALUES(1,'Introduction to Database Systems','This is an introduction to database systems...','2019-06-30','APA');
INSERT INTO "Research_Papers" VALUES(2,'Advanced Database Concepts','This article discusses advanced concepts in databases...','2020-08-27','MLA');
CREATE TABLE Review_Articles(
    id INTEGER PRIMARY KEY,
    title TEXT,
    reviewer TEXT,
    publication_date DATE,
    citation_style TEXT 
);
INSERT INTO "Review_Articles" VALUES(1,'Database Normalization Techniques','John Doe','2019-07-15','APA');
INSERT INTO "Review_Articles" VALUES(2,'Distributed Databases Management','Jane Smith','2020-09-23','MLA');
COMMIT;
