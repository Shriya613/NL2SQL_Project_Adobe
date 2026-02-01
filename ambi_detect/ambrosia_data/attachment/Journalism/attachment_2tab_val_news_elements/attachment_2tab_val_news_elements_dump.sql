BEGIN TRANSACTION;
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    editor TEXT,
    publishing_status TEXT
);
INSERT INTO "Articles" VALUES(1,'AI: The Future of Work','Discusses how AI will impact the job market and potential solutions.','Alice Johnson','Published');
INSERT INTO "Articles" VALUES(2,'Renewable Energy Solutions for a Greener Tomorrow','Explores various renewable energy sources and their effectiveness.','Bob Brown','In Review');
INSERT INTO "Articles" VALUES(3,'Cybersecurity Threats in the Modern World','Delves into common cybersecurity threats and prevention strategies.','Carol White','Pending');
INSERT INTO "Articles" VALUES(4,'Mental Health in the Age of Social Media','Analyzes the effects of social media on mental health and ways to cope.','David Lee','Published');
CREATE TABLE Headlines(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    headline TEXT,
    Author TEXT,
    publication_date DATETIME,
    category TEXT
);
INSERT INTO "Headlines" VALUES(1,'The Rise of AI','Adam Gopnik','2023-01-01 08:00:00','Technology');
INSERT INTO "Headlines" VALUES(2,'Climate Change Impact on Agriculture','Jane Doe','2023-01-02 09:00:00','Environment');
INSERT INTO "Headlines" VALUES(3,'New Laws Aimed at Curbing Crime','John Smith','2023-01-03 10:00:00','Politics');
INSERT INTO "Headlines" VALUES(4,'Health Benefits of Meditation','Emma Watson','2023-01-04 11:00:00','Wellness');
CREATE TABLE Ledes(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lede TEXT,
    Author TEXT,
    publication_date DATETIME,
    source_of_info TEXT
);
INSERT INTO "Ledes" VALUES(1,'A breakthrough in AI technology is expected to revolutionize the industry.','Adam Gopnik','2023-01-01 08:00:00','Company Press Release');
INSERT INTO "Ledes" VALUES(2,'Farmers struggle as climate change disrupts crop yields.','Jane Doe','2023-01-02 09:00:00','Government Report');
INSERT INTO "Ledes" VALUES(3,'Stricter laws aim to reduce crime rates across the country.','John Smith','2023-01-03 10:00:00','News Agency');
INSERT INTO "Ledes" VALUES(4,'Meditation has been proven to have numerous health benefits, including reduced stress and anxiety.','Emma Watson','2023-01-04 11:00:00','Research Study');
CREATE TABLE Readers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    preferred_source TEXT
);
INSERT INTO "Readers" VALUES(1,'Alex Johnson','alexj@example.com','Wired Magazine');
INSERT INTO "Readers" VALUES(2,'Samantha Williams','samwilliams@example.com','National Geographic');
INSERT INTO "Readers" VALUES(3,'Tommy Anderson','tommyanderson@example.com','Forbes');
INSERT INTO "Readers" VALUES(4,'Lucy Davis','lucydavis@example.com','Bloomberg');
INSERT INTO "Readers" VALUES(5,'Mark Taylor','marktaylor@example.com','The Economist');
CREATE TABLE Reporters(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    phone_number TEXT UNIQUE
);
INSERT INTO "Reporters" VALUES(1,'Adam Gopnik','adamg@example.com','123-456-7890');
INSERT INTO "Reporters" VALUES(2,'Jane Doe','janed@example.com','234-567-8901');
INSERT INTO "Reporters" VALUES(3,'John Smith','johns@example.com','345-678-9012');
INSERT INTO "Reporters" VALUES(4,'Emma Watson','emmaw@example.com','456-789-0123');
INSERT INTO "Reporters" VALUES(5,'Kelly Clarkson','kellyc@example.com','567-890-1234');
CREATE TABLE Sources(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    url TEXT UNIQUE
);
INSERT INTO "Sources" VALUES(1,'Wired Magazine','https://www.wired.com');
INSERT INTO "Sources" VALUES(2,'National Geographic','https://www.nationalgeographic.com');
INSERT INTO "Sources" VALUES(3,'Forbes','https://www.forbes.com');
INSERT INTO "Sources" VALUES(4,'Bloomberg','https://www.bloomberg.com');
INSERT INTO "Sources" VALUES(5,'The Economist','https://www.economist.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Headlines',4);
INSERT INTO "sqlite_sequence" VALUES('Ledes',4);
INSERT INTO "sqlite_sequence" VALUES('Articles',4);
INSERT INTO "sqlite_sequence" VALUES('Reporters',5);
INSERT INTO "sqlite_sequence" VALUES('Sources',5);
INSERT INTO "sqlite_sequence" VALUES('Readers',5);
COMMIT;
