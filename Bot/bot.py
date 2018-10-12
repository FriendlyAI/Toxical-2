#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import pickle
import time

import discord
from discord import Game
from discord.ext.commands import Bot

from lstm_network import create

NEURAL_NET = create()

BOT_PREFIX = '!'
# Get at https://discordapp.com/developers/applications/me
TOKEN = open('token.txt', 'r').readline()

client = Bot(command_prefix=BOT_PREFIX)

MAX_SCORE = 100
WARNING_SCORE = 20
BAN_SCORE = 0


def get_sentiment(sentence):
    prediction = NEURAL_NET.predict(sentence)
    negative_score = prediction[0]
    non_negative_score = prediction[1]
    string_format = f'Positive: {non_negative_score}\n' \
                    f'Negative: {negative_score}\n' \
                    f'Composite: {non_negative_score - negative_score}'
    return non_negative_score - negative_score, string_format


# class for member info
class DiscordMember:
    def __init__(self, uid, last_message_time):
        self.id = uid
        self.score = MAX_SCORE
        self.last_message_time = last_message_time

    def __eq__(self, other):
        if self.id == other.id:
            return True
        return False

    def __str__(self):
        return 'ID: ' + self.id + '\n' + \
               'Score: ' + str(self.score) + '\n' + \
               '--------------------------------'


# loads data from previous session of bot
try:
    member_list = pickle.load(open('users.pickle', 'rb'))
except (OSError, IOError) as e:
    member_list = []
    pickle.dump(member_list, open('users.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

@client.event
async def on_ready():
    await client.change_presence(game=Game(name='positively'))
    print(f'Logged in as {client.user.name}\n')
    servers = list(client.servers)
    for server in servers:
        for member in server.members:
            temp = DiscordMember(member.id, time.time())
            if temp not in member_list:
                member_list.append(temp)
    for member in member_list:
        print(member)


async def list_servers():
    await client.wait_until_ready()
    print('Current servers:')
    for server in client.servers:
        print(server.name)
    print()


@client.event
async def on_message(message):
    await client.process_commands(message)
    if message.content and message.content != '!score' and message.author.id != client.user.id:
        score_change, string_format = get_sentiment(message.content)
        score_change = score_change if score_change + 1 < 0 else 0  # Only count if score sentiment < -1
        # print(string_format)  # For testing

        # Update score

        current_time = time.time()

        temp = DiscordMember(message.author.id, time.time())
        if temp not in member_list:
            member_list.append(temp)

        for user in member_list:
            if user.id == message.author.id:
                prev_score = user.score
                old_time = user.last_message_time

                time_points = (current_time - old_time) / 600

                new_score = min(prev_score + time_points, MAX_SCORE) + score_change

                user.score = max(new_score, 0)
                user.last_message_time = current_time
                pickle.dump(member_list, open('users.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

                if new_score <= BAN_SCORE:
                    try:
                        await client.ban(message.server.get_member(message.author.id), delete_message_days=0)
                    except discord.errors.Forbidden:
                        print('Privilege too low')
                    else:
                        member_list.remove(temp)

                elif new_score <= WARNING_SCORE:
                    await client.send_message(message.channel,
                                              f'**WARNING <@{message.author.id}> your positivity score is very low '
                                              f'({"{0:0.1f}".format(new_score)}/{MAX_SCORE})**'
                                              f'\nYou will be banned if your score reaches {BAN_SCORE}.')
                break


@client.command(pass_context=True)
async def score(ctx):
    temp = DiscordMember(ctx.message.author.id, time.time())
    if temp not in member_list:
        member_list.append(temp)

    current_time = time.time()

    for user in member_list:
        if user.id == ctx.message.author.id:
            prev_score = user.score
            old_time = user.last_message_time

            time_points = (current_time - old_time) / 600

            user.score = min(prev_score + time_points, MAX_SCORE)
            user.last_message_time = current_time
            pickle.dump(member_list, open('users.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            await client.send_message(ctx.message.channel,
                                      f'{ctx.message.author}\'s score is '
                                      f'{"{0:0.1f}".format(min(prev_score + time_points, MAX_SCORE))}/{MAX_SCORE}')

if __name__ == '__main__':
    client.loop.create_task(list_servers())
    client.run(TOKEN)
