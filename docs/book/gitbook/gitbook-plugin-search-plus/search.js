require([
    'gitbook',
    'jquery'
], function (gitbook, $) {
    const MAX_DESCRIPTION_SIZE = 500
    let INDEX_DATA = {}
    const usePushState = (typeof window.history.pushState !== 'undefined')

    // DOM Elements
    const $body = $('body')

    // Throttle search
    function throttle(fn, wait) {
        var timeout

        return function () {
            var ctx = this
            var args = arguments
            if (!timeout) {
                timeout = setTimeout(function () {
                    timeout = null
                    fn.apply(ctx, args)
                }, wait)
            }
        }
    }

    function displayResults(res) {
        const $bookSearchResults = $('#book-search-results');
        const $searchList = $bookSearchResults.find('.search-results-list');
        const $searchTitle = $bookSearchResults.find('.search-results-title');
        const $searchResultsCount = $searchTitle.find('.search-results-count');
        const $searchQuery = $searchTitle.find('.search-query');

        $bookSearchResults.addClass('open');

        const noResults = res.count === 0;
        $bookSearchResults.toggleClass('no-results', noResults);

        // Clear old results
        $searchList.empty();

        // Display title for research
        $searchResultsCount.text(res.count);
        $searchQuery.text(res.query);

        // Create an <li> element for each result
        let seq = 1;
        res.results.forEach(function (item) {
            const $li = $('<li>', {
                'class': 'search-results-item'
            });

            const $title = $('<h3>')
            $title.prepend($('<span>', {
                'text' : '[' + seq + '] '
            }));
            seq++;

            const $link = $('<a>', {
                'href': gitbook.state.basePath + '/' + item.url + '?h=' + encodeURIComponent(res.query),
                'text': item.title,
                'data-is-search': 1
            });

            if ($link[0].href.split('?')[0] === window.location.href.split('?')[0]) {
                $link[0].setAttribute('data-need-reload', 1);
            }

            let content = item.body.trim();
            if (content.length > MAX_DESCRIPTION_SIZE) {
                content = content + '...';
            }
            const $content = $('<p>').html(content);

            $title.append($link);
            $li.append($title);
            $li.append($content);
            $searchList.append($li);
        });
        $('.body-inner').scrollTop(0);
    }

    function escapeRegExp(keyword) {
        // escape regexp prevserve word
        return String(keyword).replace(/([-.*+?^${}()|[\]/\\])/g, '\\$1');
    }

    function query(searchString) {
        if (!searchString) {
            return;
        } else if (typeof searchString === 'string' && searchString.trim() === "") {
            return;
        }

        const searchReg = new RegExp(escapeRegExp(searchString), 'i');
        const replaceReg = new RegExp('(' + escapeRegExp(searchString) + ')', 'gi');

        const results = []
        for (let i in INDEX_DATA) {
            const store = INDEX_DATA[i]
            const index = store.body.search(searchReg);
            if (index !== -1) {
                results.push({
                    url: store.url,
                    title: store.title,
                    body: formatText(
                        store.body.substr(Math.max(0, index - 50), MAX_DESCRIPTION_SIZE),
                        replaceReg)
                });
            }
        }
        displayResults({
            count: results.length,
            query: searchString,
            results: results
        });
    }

    function formatText(text, reg) {
        let result = text.replace(reg, '\x7f$1\x7f');
        result = escapeHTML(result);
        result =  result.replace(
            /\x7f([^\x7f]+)\x7f/g,
            '<span class="search-highlight-keyword">' + escapeHTML('$1') + '</span>');
        return result;
    }

    function escapeHTML(string) {
        if (typeof string !== 'string') {
            return string;
        }
        return string.replace(/[&'`"<>]/g, function (match) {
            return {
                '&': '&amp;',
                "'": '&#x27;',
                '`': '&#x60;',
                '"': '&quot;',
                '<': '&lt;',
                '>': '&gt;',
            } [match]
        });
    }

    function launchSearch(keyword) {
        // Add class for loading
        $body.addClass('with-search');
        $body.addClass('search-loading');

        function doSearch() {
            query(keyword);
            $body.removeClass('search-loading');
        }

        throttle(doSearch)();
    }

    function closeSearch() {
        $body.removeClass('with-search');
        $('#book-search-results').removeClass('open');
    }

    function bindSearch() {
        // Launch query based on input content
        function handleUpdate() {
            const $searchInput = $('#book-search-input input');
            const keyword = $searchInput.val();

            if (keyword.length === 0) {
                closeSearch();
            } else {
                launchSearch(keyword);
            }
        }

        $body.on('keyup', '#book-search-input input', function (e) {
            if (e.keyCode === 13) {
                if (usePushState) {
                    const uri = updateQueryString('q', $(this).val())
                    window.history.pushState({
                        path: uri
                    }, null, uri);
                }
            }
            handleUpdate();
        })

        // Push to history on blur
        $body.on('blur', '#book-search-input input', function (e) {
            // Update history state
            if (usePushState) {
                const uri = updateQueryString('q', $(this).val());
                window.history.pushState({
                    path: uri
                }, null, uri);
            }
        })
    }

    gitbook.events.on('start', function () {
        bindSearch()
        $.getJSON(gitbook.state.basePath + '/search_plus_index.json').then(function (data) {
            INDEX_DATA = data;
            showResult();
            closeSearch();
        });
    });

    const markConfig = {
        'ignoreJoiners': true,
        'acrossElements': true,
        'separateWordSearch': false
    };
    // highlight
    function highLightPageInner(keyword) {
        const pageInner = $('.page-inner');
        if (/(?:(.+)?\:)(.+)/.test(keyword)) {
            pageInner.mark(RegExp.$1, markConfig);
        }
        pageInner.mark(keyword, markConfig);

        setTimeout(function () {
            const mark = $('mark[data-markjs="true"]');
            if (mark.length) {
                mark[0].scrollIntoView();
            }
        }, 100);
    }

    function showResult() {
        if (/\b(q|h)=([^&]+)/.test(window.location.search)) {
            const type = RegExp.$1;
            const keyword = decodeURIComponent(RegExp.$2);
            if (type === 'q') {
                launchSearch(keyword);
            } else {
                highLightPageInner(keyword);
            }
            $('#book-search-input input').val(keyword);
        }
    }

    gitbook.events.on('page.change', showResult);

    function updateQueryString(key, value) {
        value = encodeURIComponent(value);

        let url = window.location.href.replace(/([?&])(?:q|h)=([^&]+)(&|$)/, function (all, pre, value, end) {
            if (end === '&') {
                return pre;
            }
            return '';
        });
        const re = new RegExp('([?&])' + key + '=.*?(&|#|$)(.*)', 'gi');

        if (re.test(url)) {
            if (typeof value !== 'undefined' && value !== null) {
                return url.replace(re, '$1' + key + '=' + value + '$2$3');
            } else {
                const hash = url.split('#');
                url = hash[0].replace(re, '$1$3').replace(/(&|\?)$/, '');
                if (typeof hash[1] !== 'undefined' && hash[1] !== null) {
                    url += '#' + hash[1];
                }
                return url;
            }
        } else {
            if (typeof value !== 'undefined' && value !== null) {
                const separator = url.indexOf('?') !== -1 ? '&' : '?';
                const hash = url.split('#');
                url = hash[0] + separator + key + '=' + value;
                if (typeof hash[1] !== 'undefined' && hash[1] !== null) {
                    url += '#' + hash[1];
                }
                return url;
            } else {
                return url;
            }
        }
    }
    window.addEventListener('click', function (e) {
        if (e.target.tagName === 'A' && e.target.getAttribute('data-need-reload')) {
            setTimeout(function () {
                window.location.reload();
            }, 100);
        }
    }, true);
})
