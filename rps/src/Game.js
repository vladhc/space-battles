import React, {useState, useReducer} from 'react';
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome';
import {
    faChevronLeft,
    faChevronRight,
    faPlay,
    faPauseCircle
} from '@fortawesome/free-solid-svg-icons';

import {Table, Frame, Button, Loading, Project, Words, Link, withStyles } from "@arwes/arwes";
import {Container} from "./components/container";
import {useHistory, useParams} from "react-router";
import {useInterval} from "./hooks/useInterval";
import {GameGraph} from "./GameGraph";
import {GameHeader} from "./GameHeader";
import {GameCanvas} from "./GameCanvas";


export const BASE_URL = "http://localhost:8080";

const styles = themes => {
    console.log(themes);
    return {
        frames: {
            marginTop: themes.margin,
        },
        header: {
            paddingLeft: themes.padding
        },
        frameContent: {
            padding: themes.padding,
        },
        heading: {
            "@global": {
                "h1": {
                    width:"100%",
                    "@global": {
                        span: {
                            width: "100%",
                        }
                    }
                }
            },
            width: "100%",
        },
        controls: {
            display: "flex",
            justifyContent: "center",
            marginTop: themes.margin
        },

        controlElements: {
            marginLeft: themes.margin,
            marginRight: themes.margin,
            cursor: "pointer"
        }

    };
};

const computeRoundData = (round) => {
    const ships      = [[0,0,0],[0,0,0],[0,0,0]];
    const planets    = [0,0,0];
    const fleets     = [0,0,0];
    const production = [[0,0,0],[0,0,0],[0,0,0]];
    const add = (a,b) => {
        a[0] += b[0];
        a[1] += b[1];
        a[2] += b[2];
    };

    for (let fi = 0; fi < round.fleets.length; fi++) {
        const fleet = round.fleets[fi];
        fleets[fleet.owner_id] += 1;
        add(ships[fleet.owner_id], fleet.ships);
    }
    for (let pi = 0; pi < round.planets.length; pi++) {
        const planet = round.planets[pi];
        planets[planet.owner_id] += 1;
        add(ships[planet.owner_id], planet.ships);
        add(production[planet.owner_id], planet.production);
    }

    return [
        round["game_over"] ? "Final Standings" : round.round,
        ships[0][0] + ships[0][1] + ships[0][2] + ships[1][0] + ships[1][1] + ships[1][2] + ships[2][0] + ships[2][1] + ships[2][2],
        planets[1], planets[2],
        ships[1].join(","), ships[2].join(","),
        production[1].join(","), production[2].join(","),
        fleets[1],fleets[2]
    ];
};

const reducer = (state, action) => {
    switch (action.type) {
        case 'incrementMove':
            if (!state.game || !state.game.length) return state;
            return {...state, turn: Math.min(state.game.length-1, state.turn + 1)};
        case 'setMove': {
            return {...state, turn: action.value};
        }
        case 'decrementMove':
            return {...state, turn: Math.max(0, state.turn - 1), playback: false};
        case 'setMove':
            return {...state, turn: action.value, playback: false};
        case 'togglePlayback': {
            return {...state, playback: !state.playback};
        }
        case 'setPlayback': {
            return {...state, playback: action.value};
        }
        case 'updateGame': {
            return {...state, game: [...(state.game || []), ...action.value]};
        }
        case 'setGameId': {
            console.log("setting gameId", action);
            return {...state, gameId: action.value, game: undefined};
        }
        default:
            throw Error();
    }
};

export const Game = withStyles(styles)(({show, classes, showLatest}) => {
    const {id, playerName} = useParams();

    const [{turn, playback, game, gameId}, dispatch] = useReducer(reducer, {turn: 0, playback: false, game: undefined, gameId: id});
    const history = useHistory();
    const [info, setInfo] = useState(undefined);

    useInterval(async () => {
        const data = await fetch(`${BASE_URL}/player/${playerName}/latest_game.json`) ;
        const json = await data.json();

        if (gameId !== json.last && json.last && (!game || !turn || (game.length && game[turn]["game_over"]) || !playback)) {
            dispatch({type: 'setGameId', value: json.last});
            dispatch({type: 'setPlayback', value: true});
            dispatch({type: 'setMove', value: 0});
            setInfo(null);
        }
    }, playerName === undefined ? null : 3000);

    useInterval(async () => {
        const data = await fetch(`${BASE_URL}/info.json`) ;
        const json = await data.json();

        console.log(json);
        const lastGameId = json && json.last_games && json.last_games.length && json.last_games[0] &&  json.last_games[0].game_id;

        if (gameId !== lastGameId && lastGameId && (!game || !turn || (game.length && game[turn]["game_over"]) || !playback)) {
            dispatch({type: 'setGameId', value: lastGameId});
            dispatch({type: 'setPlayback', value: true});
            dispatch({type: 'setMove', value: 0});
            setInfo(null);
        }
    }, !showLatest ? null : 3000);

    useInterval(async () => {
        if (gameId !== undefined) {
            const data = await fetch(`${BASE_URL}/game/${gameId}/info.json`);
            const json = await data.json();
            setInfo(json);
        }
    }, info && info.finished ? null : 1000);

    useInterval(async () => {
        if (gameId !== undefined) {
            const data = await fetch(`${BASE_URL}/game/${gameId}/rounds/${game && game.length || 0}`);
            const json = await data.json();
            dispatch({type: 'updateGame', value: json});
        }
    }, game && game.length && game[game.length -1]['game_over'] ? null : 500);


    useInterval(() => playback && dispatch({type: 'incrementMove'}), 33);

    return (
        <Container>
            <Frame
                animate={true}
                level={3}
                corners={4}
                show={show}
                layer='primary'>
                {(anim) => <GameHeader playerName={playerName} className={classes.header} info={info} history={history} show={anim.entered} arwesShow={show}/>}
            </Frame>

            <Frame
                animate={true}
                level={3}
                corners={4}
                show={show}
                layer='primary'>
                {(anim) =>
                    <div className={classes.frameContent}>
                        {show && <div className={classes.controls}>
                            <Button animate layer='primary' onClick={() => dispatch({type: 'decrementMove'})} show={anim.entered}>
                                {anim.entered && <FontAwesomeIcon className={classes.controlElements} icon={faChevronLeft} size="lg"/>}
                            </Button>
                            <Button animate layer='primary' onClick={() => dispatch({type: 'togglePlayback'})} show={anim.entered}>
                                {anim.entered && <FontAwesomeIcon fixedWidth className={classes.controlElements} icon={playback ? faPauseCircle : faPlay} size="lg"/>}
                            </Button>
                            <Button animate layer='primary' onClick={() => dispatch({type: 'incrementMove'})} show={anim.entered}>
                                {anim.entered && <FontAwesomeIcon className={classes.controlElements} icon={faChevronRight} size="lg"/>}
                            </Button>
                        </div>}
                        <GameGraph game={game} turn={turn} dispatch={dispatch} show={anim.entered} arwesShow={show}/>
                    </div>
                }
            </Frame>

            <Frame
                className={classes.frames}
                show={show}
                animate={true}
                level={3}
                corners={4}
                layer='primary'>
                {(anim) => anim.entered && (game ?
                        <GameCanvas turn={game[turn]} info={info} gameId={gameId}/> :
                        show && <Loading animate/> || null
                )}
            </Frame>

            <Frame
                className={classes.frames}
                show={show}
                animate
                level={3}
                corners={4}
                layer='primary'>
                {(anim) => anim.entered && (info && game && game[turn] ?
                        <div className={classes.frameContent}>
                            <Table animate
                                   headers={[
                                       "Round",
                                       "Total Fleets",
                                       <Words layer="success"><div>Planets</div>{`${info["player1"]}`}</Words>,
                                       <Words layer="alert"><div>Planets</div>{`${info["player2"]}`}</Words>,
                                       <Words layer="success"><div>Ships</div>{`${info["player1"]}`}</Words>,
                                       <Words layer="alert"><div>Ships</div>{`${info["player2"]}`}</Words>,
                                       <Words layer="success"><div>Production</div>{`${info["player1"]}`}</Words>,
                                       <Words layer="alert"><div>Production</div>{`${info["player2"]}`}</Words>,
                                       <Words layer="success"><div>Fleets</div>{`${info["player1"]}`}</Words>,
                                       <Words layer="alert"><div>Fleets</div>{`${info["player2"]}`}</Words>,
                                   ]}
                                   dataset={[computeRoundData(game[turn])]}/>
                        </div>:
                        show && <Loading animate/> || null
                )
                }
            </Frame>

        </Container>
    );
});