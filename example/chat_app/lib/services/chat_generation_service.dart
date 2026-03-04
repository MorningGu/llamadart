import 'dart:async';

import 'package:llamadart/llamadart.dart';

import '../models/chat_settings.dart';

class GenerationStreamUpdate {
  final String cleanText;
  final String fullThinking;
  final bool shouldNotify;
  final int generatedTokenDelta;

  const GenerationStreamUpdate({
    required this.cleanText,
    required this.fullThinking,
    required this.shouldNotify,
    this.generatedTokenDelta = 0,
  });
}

class GenerationStreamResult {
  final String fullResponse;
  final String fullThinking;
  final int generatedTokens;
  final int? firstTokenLatencyMs;
  final int elapsedMs;
  final int decodeElapsedMs;

  const GenerationStreamResult({
    required this.fullResponse,
    required this.fullThinking,
    required this.generatedTokens,
    required this.firstTokenLatencyMs,
    required this.elapsedMs,
    required this.decodeElapsedMs,
  });
}

class ChatGenerationService {
  const ChatGenerationService();

  static const int _streamRevealIntervalMs = 16;
  static const int _streamFlushBudgetMs = 220;
  static const int _tokenDeltaFlushBatchSize = 8;

  GenerationParams buildParams(ChatSettings settings) {
    return GenerationParams(
      maxTokens: settings.maxTokens,
      temp: settings.temperature,
      topK: settings.topK,
      topP: settings.topP,
      minP: settings.minP,
      penalty: settings.penalty,
      stopSequences: const <String>[],
    );
  }

  List<LlamaContentPart> buildChatParts({
    required String text,
    List<LlamaContentPart>? stagedParts,
  }) {
    return <LlamaContentPart>[
      ...?stagedParts,
      if (text.isNotEmpty) LlamaTextContent(text),
    ];
  }

  Future<GenerationStreamResult> consumeStream({
    required Stream<LlamaCompletionChunk> stream,
    required bool thinkingEnabled,
    required int uiNotifyIntervalMs,
    required String Function(String) cleanResponse,
    required bool Function() shouldContinue,
    required void Function(GenerationStreamUpdate update) onUpdate,
    Duration? stallTimeout,
  }) async {
    final stopwatch = Stopwatch()..start();

    var fullResponse = '';
    var fullThinking = '';
    var visibleCleanText = '';
    var cleanTarget = '';
    var generatedTokens = 0;
    var sawFirstToken = false;
    int? firstTokenLatencyMs;

    final effectiveNotifyIntervalMs = uiNotifyIntervalMs <= 0
        ? 0
        : uiNotifyIntervalMs;
    var lastUpdateAt = DateTime.fromMillisecondsSinceEpoch(0);
    var lastNotifiedCleanText = '';
    var lastNotifiedThinking = '';
    var streamCompleted = false;
    var streamCancelled = false;
    var pendingTokenDelta = 0;
    var streamElapsedMs = 0;

    void emitUpdate({bool forceNotify = false, bool flushTokenDelta = false}) {
      final now = DateTime.now();
      final hasVisibleDelta =
          visibleCleanText != lastNotifiedCleanText ||
          fullThinking != lastNotifiedThinking;

      final shouldNotify =
          forceNotify ||
          (hasVisibleDelta &&
              (effectiveNotifyIntervalMs == 0 ||
                  now.difference(lastUpdateAt).inMilliseconds >=
                      effectiveNotifyIntervalMs));

      final shouldFlushTokenDelta =
          pendingTokenDelta > 0 &&
          (forceNotify ||
              flushTokenDelta ||
              shouldNotify ||
              pendingTokenDelta >= _tokenDeltaFlushBatchSize);

      if (!shouldNotify && !shouldFlushTokenDelta) {
        return;
      }

      if (shouldNotify) {
        lastUpdateAt = now;
        lastNotifiedCleanText = visibleCleanText;
        lastNotifiedThinking = fullThinking;
      }

      final generatedTokenDelta = shouldFlushTokenDelta ? pendingTokenDelta : 0;
      if (shouldFlushTokenDelta) {
        pendingTokenDelta = 0;
      }

      onUpdate(
        GenerationStreamUpdate(
          cleanText: visibleCleanText,
          fullThinking: fullThinking,
          shouldNotify: shouldNotify,
          generatedTokenDelta: generatedTokenDelta,
        ),
      );
    }

    void advanceVisibleTextAndEmit() {
      if (!shouldContinue()) {
        streamCancelled = true;
        return;
      }

      final nextVisible = _advanceVisibleText(
        currentText: visibleCleanText,
        targetText: cleanTarget,
      );
      if (nextVisible == visibleCleanText) {
        return;
      }

      visibleCleanText = nextVisible;
      emitUpdate();
    }

    final revealTicker =
        Stream<void>.periodic(
          const Duration(milliseconds: _streamRevealIntervalMs),
          (_) {},
        ).listen((_) {
          if (streamCompleted || streamCancelled) {
            return;
          }
          advanceVisibleTextAndEmit();
        });

    try {
      final effectiveStream = stallTimeout == null
          ? stream
          : stream.timeout(
              stallTimeout,
              onTimeout: (sink) {
                sink.addError(
                  TimeoutException(
                    'Generation stalled waiting for output.',
                    stallTimeout,
                  ),
                );
              },
            );

      await for (final chunk in effectiveStream) {
        if (!shouldContinue()) {
          streamCancelled = true;
          break;
        }

        final delta = chunk.choices.first.delta;
        final content = delta.content ?? '';
        final thinking = thinkingEnabled ? (delta.thinking ?? '') : '';

        if (!sawFirstToken &&
            (content.isNotEmpty ||
                thinking.isNotEmpty ||
                (delta.toolCalls?.isNotEmpty ?? false))) {
          firstTokenLatencyMs = stopwatch.elapsedMilliseconds;
          sawFirstToken = true;
        }

        fullResponse += content;
        fullThinking += thinking
            .replaceAll(r'\n', '\n')
            .replaceAll(r'\r', '\r');
        generatedTokens++;
        pendingTokenDelta += 1;

        cleanTarget = cleanResponse(fullResponse);
        visibleCleanText = _advanceVisibleText(
          currentText: visibleCleanText,
          targetText: cleanTarget,
        );

        emitUpdate();
      }

      streamCompleted = true;
      streamElapsedMs = stopwatch.elapsedMilliseconds;

      if (!streamCancelled && visibleCleanText != cleanTarget) {
        final flushDeadline = DateTime.now().add(
          const Duration(milliseconds: _streamFlushBudgetMs),
        );
        while (visibleCleanText != cleanTarget &&
            DateTime.now().isBefore(flushDeadline)) {
          advanceVisibleTextAndEmit();
          if (visibleCleanText == cleanTarget) {
            break;
          }
          await Future<void>.delayed(
            const Duration(milliseconds: _streamRevealIntervalMs),
          );
        }
      }

      if (!streamCancelled) {
        if (visibleCleanText != cleanTarget) {
          visibleCleanText = cleanTarget;
          emitUpdate(forceNotify: true, flushTokenDelta: true);
        } else if (visibleCleanText != lastNotifiedCleanText ||
            fullThinking != lastNotifiedThinking) {
          emitUpdate(forceNotify: true, flushTokenDelta: true);
        } else if (pendingTokenDelta > 0) {
          emitUpdate(flushTokenDelta: true);
        }
      }
    } finally {
      await revealTicker.cancel();
    }

    stopwatch.stop();
    if (streamElapsedMs <= 0) {
      streamElapsedMs = stopwatch.elapsedMilliseconds;
    }
    final safeFirstTokenLatencyMs = firstTokenLatencyMs ?? 0;
    final decodeElapsedMs = streamElapsedMs > safeFirstTokenLatencyMs
        ? streamElapsedMs - safeFirstTokenLatencyMs
        : 0;

    return GenerationStreamResult(
      fullResponse: fullResponse,
      fullThinking: fullThinking,
      generatedTokens: generatedTokens,
      firstTokenLatencyMs: firstTokenLatencyMs,
      elapsedMs: streamElapsedMs,
      decodeElapsedMs: decodeElapsedMs,
    );
  }

  String _advanceVisibleText({
    required String currentText,
    required String targetText,
  }) {
    if (currentText == targetText) {
      return targetText;
    }

    final canPrefixAdvance =
        targetText.length > currentText.length &&
        targetText.startsWith(currentText);
    if (!canPrefixAdvance) {
      return targetText;
    }

    final backlog = targetText.length - currentText.length;
    final revealStep = _revealStepForBacklog(backlog);
    var nextLength = currentText.length + revealStep;
    if (nextLength >= targetText.length) {
      return targetText;
    }

    nextLength = _alignToUtf16Boundary(targetText, nextLength);
    if (nextLength >= targetText.length) {
      return targetText;
    }

    return targetText.substring(0, nextLength);
  }

  int _alignToUtf16Boundary(String text, int end) {
    if (end <= 0 || end >= text.length) {
      return end;
    }

    final previousCodeUnit = text.codeUnitAt(end - 1);
    final nextCodeUnit = text.codeUnitAt(end);
    if (_isLeadingSurrogate(previousCodeUnit) &&
        _isTrailingSurrogate(nextCodeUnit)) {
      return end + 1;
    }

    return end;
  }

  bool _isLeadingSurrogate(int codeUnit) {
    return codeUnit >= 0xD800 && codeUnit <= 0xDBFF;
  }

  bool _isTrailingSurrogate(int codeUnit) {
    return codeUnit >= 0xDC00 && codeUnit <= 0xDFFF;
  }

  int _revealStepForBacklog(int backlog) {
    return backlog > 0 ? 1 : 0;
  }
}
