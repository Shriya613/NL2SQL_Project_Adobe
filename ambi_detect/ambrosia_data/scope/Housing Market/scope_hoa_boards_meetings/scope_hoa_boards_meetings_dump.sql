BEGIN TRANSACTION;
CREATE TABLE Hoa_Board_Members (
    hoa_board_id INTEGER,
    member_id INTEGER,
    FOREIGN KEY (hoa_board_id) REFERENCES Hoa_Boards(board_id),
    FOREIGN KEY (member_id) REFERENCES Members(member_id),
    PRIMARY KEY (hoa_board_id, member_id)
);
INSERT INTO "Hoa_Board_Members" VALUES(1,1);
INSERT INTO "Hoa_Board_Members" VALUES(1,2);
INSERT INTO "Hoa_Board_Members" VALUES(2,3);
INSERT INTO "Hoa_Board_Members" VALUES(2,4);
INSERT INTO "Hoa_Board_Members" VALUES(3,5);
CREATE TABLE Hoa_Boards (
    board_id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT,
    president_name TEXT
);
INSERT INTO "Hoa_Boards" VALUES(1,'123 Main St','John Doe');
INSERT INTO "Hoa_Boards" VALUES(2,'456 Elm St','Jane Smith');
INSERT INTO "Hoa_Boards" VALUES(3,'789 Oak St','Michael Johnson');
INSERT INTO "Hoa_Boards" VALUES(4,'001 Pine St','Emily Davis');
INSERT INTO "Hoa_Boards" VALUES(5,'002 Fir St','David Brown');
CREATE TABLE Hoa_Boards_Meetings (
    hoa_board_id INTEGER,
    meeting_id INTEGER,
    FOREIGN KEY (hoa_board_id) REFERENCES Hoa_Boards(board_id),
    FOREIGN KEY (meeting_id) REFERENCES Meetings(meeting_id),
    PRIMARY KEY (hoa_board_id, meeting_id)
);
INSERT INTO "Hoa_Boards_Meetings" VALUES(1,1);
INSERT INTO "Hoa_Boards_Meetings" VALUES(1,2);
INSERT INTO "Hoa_Boards_Meetings" VALUES(2,3);
INSERT INTO "Hoa_Boards_Meetings" VALUES(2,4);
INSERT INTO "Hoa_Boards_Meetings" VALUES(3,5);
INSERT INTO "Hoa_Boards_Meetings" VALUES(2,1);
INSERT INTO "Hoa_Boards_Meetings" VALUES(3,1);
INSERT INTO "Hoa_Boards_Meetings" VALUES(4,1);
INSERT INTO "Hoa_Boards_Meetings" VALUES(5,1);
CREATE TABLE Meeting_Attendees (
    meeting_id INTEGER,
    attendee_id INTEGER,
    FOREIGN KEY (meeting_id) REFERENCES Meetings(meeting_id),
    FOREIGN KEY (attendee_id) REFERENCES Members(member_id),
    PRIMARY KEY (meeting_id, attendee_id)
);
INSERT INTO "Meeting_Attendees" VALUES(1,1);
INSERT INTO "Meeting_Attendees" VALUES(1,2);
INSERT INTO "Meeting_Attendees" VALUES(2,3);
INSERT INTO "Meeting_Attendees" VALUES(3,4);
INSERT INTO "Meeting_Attendees" VALUES(4,5);
CREATE TABLE Meetings (
    meeting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_type TEXT 
);
INSERT INTO "Meetings" VALUES(1,'Annual_Meeting');
INSERT INTO "Meetings" VALUES(2,'Special_Meeting');
INSERT INTO "Meetings" VALUES(3,'Budget_Meeting');
INSERT INTO "Meetings" VALUES(4,'Maintenance_Meeting');
INSERT INTO "Meetings" VALUES(5,'Election_Meeting');
CREATE TABLE Members (
    member_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Members" VALUES(1,'Alice Green','alice@example.com');
INSERT INTO "Members" VALUES(2,'Bob White','bob@example.com');
INSERT INTO "Members" VALUES(3,'Carol Black','carol@example.com');
INSERT INTO "Members" VALUES(4,'Dave Blue','dave@example.com');
INSERT INTO "Members" VALUES(5,'Elizabeth Red','elizabeth@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Hoa_Boards',5);
INSERT INTO "sqlite_sequence" VALUES('Meetings',5);
INSERT INTO "sqlite_sequence" VALUES('Members',5);
COMMIT;
